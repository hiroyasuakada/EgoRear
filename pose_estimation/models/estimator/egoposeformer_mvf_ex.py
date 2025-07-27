import copy

import torch
import torch.nn as nn

from pose_estimation.models.estimator import EgoPoseFormerHeatmapMVFEX
from pose_estimation.models.utils.deform_attn import MSDeformAttn
from pose_estimation.models.utils.transformer import CustomMultiheadAttention, FFN

from pose_estimation.utils.camera_models import projection_funcs, FishEyeCameraCalibratedModel

from einops import rearrange
from torch._dynamo import allow_in_graph
allow_in_graph(rearrange)

from loguru import logger

INF = 1e10
EPS = 1e-6


class EgoPoseFormerMVFEX(nn.Module):
    def __init__(
        self,
        num_views,
        image_size,
        camera_model,
        heatmap_mvf_cfg,
        pose3d_cfg,
        **kwargs
    ):
        super(EgoPoseFormerMVFEX, self).__init__(**kwargs)

        heatmap_mvf_cfg.update({
            "num_views": num_views,
            "image_size": image_size,
            "camera_model": camera_model,
        })
        self.heatmap_estimator = EgoPoseFormerHeatmapMVFEX(**heatmap_mvf_cfg)
        self.use_pred_heatmap_init = self.heatmap_estimator.use_pred_heatmap_init

        pose3d_cfg.update({
            "num_views": num_views,
            "image_size": image_size,
            "use_pred_heatmap_init": self.use_pred_heatmap_init,
            "camera_model": camera_model,
        })
        self.pose3d_estimator = EgoPoseFormerPose3D(**pose3d_cfg)

    def forward(self, img, coord_trans_mat=None, origin_3d=None):
        list_pred_heatmap, list_frame_feat = self.heatmap_estimator(img)
        pred_heatmap_final = list_pred_heatmap[-1]

        frame_feat_init = list_frame_feat[0]
        frame_feat_final = list_frame_feat[-1]

        list_pred_pose3d = self.pose3d_estimator(frame_feat_init, frame_feat_final, pred_heatmap_final, coord_trans_mat, origin_3d)

        return list_pred_pose3d, list_pred_heatmap


class EgoPoseFormerPose3D(nn.Module):
    def __init__(
        self,
        num_views,
        image_size,
        use_pred_heatmap_init,
        num_joints,
        input_dims,
        embed_dims,
        mlp_dims,
        mlp_dropout,
        num_mlp_layers,
        transformer_cfg,
        num_former_layers,
        num_pred_mlp_layers,
        camera_model,
        feat_down_stride,
        coor_norm_max,
        coor_norm_min,
        conv_heatmap_dim_init,
        norm_mlp_pred=False,
        use_mlp_avgpool=True,
        use_mlp_heatmap=False,
        camera_calib_file_dir_path=None,
        **kwargs
    ):
        super(EgoPoseFormerPose3D, self).__init__(**kwargs)

        self.invalid_pad = INF

        self.num_views = num_views
        self.num_joints = num_joints
        self.embed_dims = embed_dims

        self.feat_down_stride = feat_down_stride
        self.feat_shape = (
            image_size[0] // feat_down_stride,
            image_size[1] // feat_down_stride,
        )
        self.image_size = image_size
        self.camera_model = camera_model

        self.use_pred_heatmap_init = use_pred_heatmap_init


        # Camera model
        if (self.camera_model == "ego4view_syn") or (self.camera_model == "ego4view_rw"):
            self.camera_front_left_model = FishEyeCameraCalibratedModel(
                camera_model, camera_calib_file_dir_path, "camera_front_left"
            )
            self.camera_front_right_model = FishEyeCameraCalibratedModel(
                camera_model, camera_calib_file_dir_path, "camera_front_right"
            )
            self.camera_back_left_model = FishEyeCameraCalibratedModel(
                camera_model, camera_calib_file_dir_path, "camera_back_left"
            )
            self.camera_back_right_model = FishEyeCameraCalibratedModel(
                camera_model, camera_calib_file_dir_path, "camera_back_right"
            )

            assert num_views == 4

        elif (self.camera_model == "ego4view_syn_stereo_front") or (self.camera_model == "ego4view_rw_stereo_front"):
            self.camera_front_left_model = FishEyeCameraCalibratedModel(
                camera_model, camera_calib_file_dir_path, "camera_front_left"
            )
            self.camera_front_right_model = FishEyeCameraCalibratedModel(
                camera_model, camera_calib_file_dir_path, "camera_front_right"
            )

            assert num_views == 2

        elif (self.camera_model == "ego4view_syn_stereo_back") or (self.camera_model == "ego4view_rw_stereo_back"):
            self.camera_back_left_model = FishEyeCameraCalibratedModel(
                camera_model, camera_calib_file_dir_path, "camera_back_left"
            )
            self.camera_back_right_model = FishEyeCameraCalibratedModel(
                camera_model, camera_calib_file_dir_path, "camera_back_right"
            )

            assert num_views == 2

        self.feat_proj = nn.Conv2d(input_dims, embed_dims, 1, 1, 0)

        self.layers = nn.ModuleList()
        for idx in range(num_former_layers):
            _cfg = copy.deepcopy(transformer_cfg)
            _cfg.update({
                "num_views": num_views,
                "embed_dims": embed_dims,
                "feat_shape": self.feat_shape,
            })
            self.layers.append(EgoPoseFormerTransformerLayer(**_cfg))

        self.query_gen_mlp = nn.Sequential(
            nn.Linear(4, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims)
        )

        self.use_mlp_avgpool = use_mlp_avgpool
        self.use_mlp_heatmap = use_mlp_heatmap

        if use_mlp_avgpool:
            mlp_layers = []
            in_dims = embed_dims * num_views
            for _ in range(num_mlp_layers):
                mlp_layers.append(
                    nn.Sequential(
                        nn.Linear(in_dims, mlp_dims),
                        nn.GELU(),
                        nn.Dropout(mlp_dropout),
                    )
                )
                in_dims = mlp_dims
            mlp_layers.append(nn.Linear(in_dims, 3 * self.num_joints))
            self.mlp_pred = nn.Sequential(*mlp_layers)
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        elif use_mlp_heatmap:
            conv_heatmap_dim_init = 32
            self.conv_heatmap_front_left = nn.Sequential(
                nn.Conv2d(15, conv_heatmap_dim_init, 3, 2, 1),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2),
                nn.Conv2d(conv_heatmap_dim_init, conv_heatmap_dim_init * 2, 3, 2, 1),
                nn.ReLU(inplace=False),
            )
            self.conv_heatmap_front_right = nn.Sequential(
                nn.Conv2d(15, conv_heatmap_dim_init, 3, 2, 1),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2),
                nn.Conv2d(conv_heatmap_dim_init, conv_heatmap_dim_init * 2, 3, 2, 1),
                nn.ReLU(inplace=False),
            )
            self.conv_frame_feat_back_left = nn.Sequential(
                nn.Conv2d(15, conv_heatmap_dim_init, 3, 2, 1),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2),
                nn.Conv2d(conv_heatmap_dim_init, conv_heatmap_dim_init * 2, 3, 2, 1),
                nn.ReLU(inplace=False),
            )
            self.conv_frame_feat_back_right = nn.Sequential(
                nn.Conv2d(15, conv_heatmap_dim_init, 3, 2, 1),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2),
                nn.Conv2d(conv_heatmap_dim_init, conv_heatmap_dim_init * 2, 3, 2, 1),
                nn.ReLU(inplace=False),
            )

            mlp_layers = []
            in_dims = num_views * conv_heatmap_dim_init * 2 * 8 * 8
            for _ in range(num_mlp_layers):
                mlp_layers.append(
                    nn.Sequential(
                        nn.Linear(in_dims, int(in_dims / 16)),
                        nn.GELU(),
                        nn.Dropout(mlp_dropout),
                    )
                )
                in_dims = int(in_dims / 16)
            mlp_layers.append(nn.Linear(in_dims, 3 * self.num_joints))
            self.mlp_pred = nn.Sequential(*mlp_layers)

        else:
            self.conv_frame_feat = nn.Sequential(
                nn.Conv2d(input_dims, int(input_dims / 2), 1, 1),
                nn.ReLU(inplace=False),
                nn.Conv2d(int(input_dims / 2), input_dims, 3, 2, 1),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2),
                nn.Conv2d(input_dims, int(input_dims / 2), 1, 1),
                nn.ReLU(inplace=False),
                nn.Conv2d(int(input_dims / 2), input_dims, 3, 2, 1),
                nn.ReLU(inplace=False),
            )

            mlp_layers = []
            in_dims = num_views * 128 * 8 * 8
            for _ in range(num_mlp_layers):
                mlp_layers.append(
                    nn.Sequential(
                        nn.Linear(in_dims, int(in_dims / 16)),
                        nn.GELU(),
                        nn.Dropout(mlp_dropout),
                    )
                )
                in_dims = int(in_dims / 16)
            mlp_layers.append(nn.Linear(in_dims, 3 * self.num_joints))
            self.mlp_pred = nn.Sequential(*mlp_layers)

        self.reg_mlp = torch.nn.ModuleList()
        for _ in range(num_former_layers):
            reg_mlp = []
            for i in range(num_pred_mlp_layers - 1):
                reg_mlp.append(nn.Linear(embed_dims, embed_dims))
                reg_mlp.append(nn.GELU())
            reg_mlp.append(nn.Linear(embed_dims, 3))
            self.reg_mlp.append(nn.Sequential(*reg_mlp))
        self.post_norm = torch.nn.ModuleList(
            [nn.LayerNorm(embed_dims) for _ in range(num_former_layers)]
        )

        self.norm_mlp_pred = norm_mlp_pred
        if norm_mlp_pred:
            self.register_buffer("coor_min", torch.tensor(coor_norm_min))
            self.register_buffer("coor_max", torch.tensor(coor_norm_max))

    def _unnorm_coor(self, coor, norm_range=(-1.0, 1.0)):
        norm_gap = (norm_range[1] - norm_range[0])
        unnormed_coor = (self.coor_max - self.coor_min) * \
                        (coor - norm_range[0]) / norm_gap + self.coor_min
        return unnormed_coor

    def _norm_coor(self, coor, norm_range=(-1.0, 1.0)):
        normed_coor = (coor - self.coor_min) / (
            self.coor_max - self.coor_min
        )
        norm_gap = (norm_range[1] - norm_range[0])
        normed_coor = norm_gap * normed_coor - norm_gap * 0.5
        return normed_coor

    def _forward_conv_mlp_heatmap(self, heatmap):
        B, V, C = heatmap.shape[:3]

        heatmap_feat_front_left = self.conv_heatmap_front_left(heatmap[:, 0, :, :, :])
        heatmap_feat_front_right = self.conv_heatmap_front_right(heatmap[:, 1, :, :, :])
        heatmap_feat_back_left = self.conv_frame_feat_back_left(heatmap[:, 2, :, :, :])
        heatmap_feat_back_right = self.conv_frame_feat_back_right(heatmap[:, 3, :, :, :])

        heatmap_feat = torch.stack(
            (
                heatmap_feat_front_left, heatmap_feat_front_right,heatmap_feat_back_left, heatmap_feat_back_right
            ),
            dim=1
        )

        x = rearrange(heatmap_feat, "b v c h w -> b (v c h w)")
        mlp_pred = self.mlp_pred(x).reshape(B, self.num_joints, 3)

        if self.norm_mlp_pred:
            self._unnorm_coor(mlp_pred)

        return mlp_pred

    def _forward_mlp_conv(self, frame_feats):

        B, V = frame_feats.shape[:2]

        frame_feats = rearrange(frame_feats, "b v c h w -> (b v) c h w")

        x = self.conv_frame_feat(frame_feats)

        x = rearrange(x, "(b v) c h w -> b (v c h w)", b=B, v=V)

        mlp_pred = self.mlp_pred(x).reshape(B, self.num_joints, 3)
        if self.norm_mlp_pred:
            self._unnorm_coor(mlp_pred)
        return mlp_pred

    def _forward_mlp(self, frame_feats):
        B, V, C = frame_feats.shape[:3]

        x = frame_feats.flatten(start_dim=0, end_dim=1)

        x = self.avg_pool(x)

        x = x.reshape(B, V * C)

        mlp_pred = self.mlp_pred(x).reshape(B, self.num_joints, 3)

        if self.norm_mlp_pred:
            self._unnorm_coor(mlp_pred)

        return mlp_pred

    def _reproject_3d_to_2d(self, init_anchors_3d, coord_trans_mat=None, origin_3d=None):

        if self.camera_model == "ego4view_syn":
            anchors_2d_front_left, anchors_valid_front_left = self.camera_front_left_model.world2camera_pytorch(init_anchors_3d)
            anchors_2d_front_right, anchors_valid_front_right = self.camera_front_right_model.world2camera_pytorch(init_anchors_3d)
            anchors_2d_back_left, anchors_valid_back_left = self.camera_back_left_model.world2camera_pytorch(init_anchors_3d)
            anchors_2d_back_right, anchors_valid_back_right = self.camera_back_right_model.world2camera_pytorch(init_anchors_3d)
            anchors_2d = torch.cat((anchors_2d_front_left, anchors_2d_front_right, anchors_2d_back_left, anchors_2d_back_right), dim=1)
            anchors_valid = torch.cat((anchors_valid_front_left, anchors_valid_front_right, anchors_valid_back_left, anchors_valid_back_right), dim=1)

        elif self.camera_model == "ego4view_syn_stereo_front":
            anchors_2d_front_left, anchors_valid_front_left = self.camera_front_left_model.world2camera_pytorch(init_anchors_3d)
            anchors_2d_front_right, anchors_valid_front_right = self.camera_front_right_model.world2camera_pytorch(init_anchors_3d)
            anchors_2d = torch.cat((anchors_2d_front_left, anchors_2d_front_right), dim=1)
            anchors_valid = torch.cat((anchors_valid_front_left, anchors_valid_front_right), dim=1)

        elif self.camera_model == "ego4view_syn_stereo_back":
            anchors_2d_back_left, anchors_valid_back_left = self.camera_back_left_model.world2camera_pytorch(init_anchors_3d)
            anchors_2d_back_right, anchors_valid_back_right = self.camera_back_right_model.world2camera_pytorch(init_anchors_3d)
            anchors_2d = torch.cat((anchors_2d_back_left, anchors_2d_back_right), dim=1)
            anchors_valid = torch.cat((anchors_valid_back_left, anchors_valid_back_right), dim=1)

        elif self.camera_model == "ego4view_rw":
            anchors_2d_front_left, anchors_valid_front_left = self.camera_front_left_model.world2camera_pytorch(init_anchors_3d, coord_trans_mat[:, 0, :, :])
            anchors_2d_front_right, anchors_valid_front_right = self.camera_front_right_model.world2camera_pytorch(init_anchors_3d, coord_trans_mat[:, 1, :, :])
            anchors_2d_back_left, anchors_valid_back_left = self.camera_back_left_model.world2camera_pytorch(init_anchors_3d, coord_trans_mat[:, 2, :, :])
            anchors_2d_back_right, anchors_valid_back_right = self.camera_back_right_model.world2camera_pytorch(init_anchors_3d, coord_trans_mat[:, 3, :, :])
            anchors_2d = torch.cat((anchors_2d_front_left, anchors_2d_front_right, anchors_2d_back_left, anchors_2d_back_right), dim=1)
            anchors_valid = torch.cat((anchors_valid_front_left, anchors_valid_front_right, anchors_valid_back_left, anchors_valid_back_right), dim=1)

        elif self.camera_model == "ego4view_rw_stereo_front":
            anchors_2d_front_left, anchors_valid_front_left = self.camera_front_left_model.world2camera_pytorch(init_anchors_3d, coord_trans_mat[:, 0, :, :])
            anchors_2d_front_right, anchors_valid_front_right = self.camera_front_right_model.world2camera_pytorch(init_anchors_3d, coord_trans_mat[:, 1, :, :])
            anchors_2d = torch.cat((anchors_2d_front_left, anchors_2d_front_right), dim=1)
            anchors_valid = torch.cat((anchors_valid_front_left, anchors_valid_front_right), dim=1)

        elif self.camera_model == "ego4view_rw_stereo_back":
            anchors_2d_back_left, anchors_valid_back_left = self.camera_back_left_model.world2camera_pytorch(init_anchors_3d, coord_trans_mat[:, 0, :, :])
            anchors_2d_back_right, anchors_valid_back_right = self.camera_back_right_model.world2camera_pytorch(init_anchors_3d, coord_trans_mat[:, 1, :, :])
            anchors_2d = torch.cat((anchors_2d_back_left, anchors_2d_back_right), dim=1)
            anchors_valid = torch.cat((anchors_valid_back_left, anchors_valid_back_right), dim=1)

        return anchors_2d, anchors_valid

    def _forward_transformer(
        self,
        image_feats,
        init_anchors_3d,
        coord_trans_mat=None,
        origin_3d=None,
    ):
        B, V, C, H, W = image_feats.shape
        J = self.num_joints

        img_feats = image_feats.permute(0, 1, 3, 4, 2).reshape(B, V, H * W, C)

        anchors_2d, anchors_valid = self._reproject_3d_to_2d(init_anchors_3d, coord_trans_mat, origin_3d)

        anchors_2d = anchors_2d.to(dtype=img_feats.dtype)

        joint_inds = (
            torch.arange(1, J+1)
            .to(dtype=img_feats.dtype, device=img_feats.device)
            .reshape(1, J, 1)
            .repeat(B, 1, 1)
        ) / float(J)
        x = self.query_gen_mlp(torch.cat((joint_inds, init_anchors_3d), dim=-1))

        preds = []
        for idx in range(len(self.layers)):
            x = self.layers[idx](
                x,
                img_feats,
                anchors_2d,
                anchors_valid,
            )
            _x = self.post_norm[idx](x)
            offset_pred = self.reg_mlp[idx](_x)
            pred = offset_pred + init_anchors_3d.detach()
            preds.append(pred)
        return preds

    def forward(self, frame_feats_init, frame_feats_final, heatmap, coord_trans_mat=None, origin_3d=None):

        if self.use_pred_heatmap_init:
            frame_feats = frame_feats_init
        else:
            frame_feats = frame_feats_final

        B, V, C, H, W = frame_feats.shape

        frame_feats = self.feat_proj(frame_feats.reshape(B * V, *frame_feats.shape[-3:]))
        frame_feats = frame_feats.reshape(B, V, *frame_feats.shape[-3:])

        if self.use_mlp_avgpool:
            mlp_pred_3d = self._forward_mlp(frame_feats_final)
        elif self.use_mlp_heatmap:
            mlp_pred_3d = self._forward_conv_mlp_heatmap(heatmap)
        else:
            mlp_pred_3d = self._forward_mlp_conv(frame_feats_final)

        init_anchors_3d = mlp_pred_3d.clone().detach()
        attn_pred_coor3ds = self._forward_transformer(
            frame_feats,
            init_anchors_3d,
            coord_trans_mat,
            origin_3d,
        )
        pred_3d_all = []
        pred_3d_all.append(mlp_pred_3d)
        pred_3d_all = pred_3d_all + attn_pred_coor3ds

        return pred_3d_all


class DeformStereoAttn(MSDeformAttn):
    def __init__(self, feat_shape, **kwargs):
        _init_cfg = {
            "d_model": kwargs.pop("embed_dim"),
            "n_heads": kwargs.pop("num_heads"),
            "n_points": 16
        }
        super(DeformStereoAttn, self).__init__(**_init_cfg)

        self.register_buffer("spatial_shapes", torch.tensor([[feat_shape[0], feat_shape[1]]], dtype=torch.long))
        self.register_buffer("start_index", torch.tensor([0,], dtype=torch.long))

    def forward(self, query, img_feat, anchors_2d):
        B, J, C = query.shape

        anchors_2d = anchors_2d.detach()

        _q = query.reshape(B, J, C)
        _kv = img_feat.reshape(B, -1, C)
        _ref_pts = anchors_2d.reshape(B, J, 1, 2)

        out = super(DeformStereoAttn, self).forward(_q, _ref_pts, _kv, self.spatial_shapes, self.start_index)
        out = out.reshape(B, J, C)
        return out


class EgoformerSpatialMHA(CustomMultiheadAttention):
    def forward(self, q, k, v, bias):
        B, J, C = q.shape

        _q = self.q_proj(q).reshape(B, J, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        _k = self.k_proj(k).reshape(B, J, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        _v = self.v_proj(v).reshape(B, J, self.num_heads, self.head_dims).permute(0, 2, 1, 3)

        attn = (_q @ _k.transpose(-2, -1)) * self.scale
        if bias is not None:
            attn = attn + bias

        attn = attn.softmax(dim=-1)

        x = (attn @ _v).permute(0, 2, 1, 3).reshape(B, J, C)
        if self.out_proj is not None:
            x = self.out_proj(x)
        return x


class EgoPoseFormerTransformerLayer(nn.Module):
    def __init__(
        self,
        num_views,
        embed_dims,
        cross_attn_cfg,
        spatial_attn_cfg,
        ffn_cfg,
        feat_shape,
    ):
        super().__init__()
        self.embed_dims = embed_dims

        _cross_attn_cfg = copy.deepcopy(cross_attn_cfg)
        _cross_attn_cfg.update({
            "embed_dim": embed_dims,
            "feat_shape": feat_shape
        })
        self.cross_attn = DeformStereoAttn(**_cross_attn_cfg)
        self.fuse_mlp = nn.Linear(num_views * embed_dims, embed_dims)
        self.norm_cross = nn.LayerNorm(embed_dims)

        _spatial_attn_cfg = copy.deepcopy(spatial_attn_cfg)
        _spatial_attn_cfg.update({"embed_dim": embed_dims})
        self.spatial_attn = EgoformerSpatialMHA(**_spatial_attn_cfg)
        self.norm_spatial = nn.LayerNorm(embed_dims)

        _ffn_cfg = copy.deepcopy(ffn_cfg)
        _ffn_cfg.update({"embed_dims": embed_dims})
        self.ffn = FFN(**_ffn_cfg)
        self.norm_ffn = nn.LayerNorm(embed_dims)

    def _run_spatial_attn(self, joint_query):
        identity = joint_query

        q = joint_query
        k = q
        v = joint_query

        attn_res = self.spatial_attn(q, k, v, bias=None)
        x = identity + attn_res
        x = self.norm_spatial(x)
        return x


    def _run_cross_attn(
        self,
        joint_query,
        image_feats,
        anchors_2d,
        anchors_valid,
    ):
        B, V, J = image_feats.shape[:3]
        identity = joint_query

        q = joint_query
        feats_per_view = []
        for i in range(V):
            qi = q
            feat_i = image_feats[:, i]
            anchors_i = anchors_2d[:, i]
            attn_res = self.cross_attn(qi, feat_i, anchors_i)
            attn_res = attn_res.masked_fill(~anchors_valid[:, i][..., None].expand_as(attn_res), 0.0)
            feats_per_view.append(attn_res)
        feats_all = self.fuse_mlp(torch.cat(feats_per_view, dim=-1))

        x = identity + feats_all
        x = self.norm_cross(x)
        return x

    def _forward_ffn(self, x):
        x = x + self.ffn(x)
        x = self.norm_ffn(x)
        return x

    def forward(
        self,
        joint_query,
        image_feats,
        anchors_2d,
        anchors_valid,
    ):
        x = joint_query

        x = self._run_cross_attn(x, image_feats, anchors_2d, anchors_valid)
        x = self._run_spatial_attn(x)
        x = self._forward_ffn(x)
        return x