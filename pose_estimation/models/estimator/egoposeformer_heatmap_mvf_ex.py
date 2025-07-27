import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pose_estimation.models.estimator import EgoPoseFormerHeatmap
from pose_estimation.models.utils.deform_attn import MSDeformAttn
from pose_estimation.models.utils.transformer import CustomMultiheadAttention, FFN

from pose_estimation.utils.loss import get_max_preds

from timm.models.layers import to_2tuple

from einops import rearrange
from torch._dynamo import allow_in_graph
allow_in_graph(rearrange)

from loguru import logger

INF = 1e10
EPS = 1e-6



class EgoPoseFormerHeatmapMVFEX(nn.Module):
    def __init__(
        self,
        num_views,
        image_size,
        num_heatmap,
        feat_down_stride,
        heatmap_threshold,
        encoder_cfg,
        mvf_cfg,
        camera_model,
        full_training=False,
        detach_heatmap_feat=False,
        detach_heatmap_feat_init=False,
        use_pred_heatmap_init=False,
        no_detach_feat_init=False,
        **kwargs
    ):
        super(EgoPoseFormerHeatmapMVFEX, self).__init__()

        self.num_views = num_views
        self.num_heatmap = num_heatmap
        self.heatmap_threshold = heatmap_threshold
        self.camera_model = camera_model
        self.full_training = full_training
        self.detach_heatmap_feat = detach_heatmap_feat
        self.detach_heatmap_feat_init = detach_heatmap_feat_init
        self.use_pred_heatmap_init = use_pred_heatmap_init
        self.no_detach_feat_init = no_detach_feat_init

        mvf_cfg = copy.deepcopy(mvf_cfg)
        mvf_cfg.update({
            "num_views": num_views,
            "num_heatmap": num_heatmap,
            "heatmap_threshold": heatmap_threshold,
            "image_size": image_size,
            "feat_down_stride": feat_down_stride,
            "detach_heatmap_feat": detach_heatmap_feat,
        })

        if num_views == 4:
            self.heatmap_estimator_stereo_front = EgoPoseFormerHeatmap(encoder_cfg, num_heatmap, detach_heatmap_feat_init)
            self.heatmap_estimator_stereo_back = EgoPoseFormerHeatmap(encoder_cfg, num_heatmap, detach_heatmap_feat_init)

            self.heatmap_refiner_front_left = HeatmapMVF(**mvf_cfg)
            self.heatmap_refiner_front_right = HeatmapMVF(**mvf_cfg)
            self.heatmap_refiner_back_left = HeatmapMVF(**mvf_cfg)
            self.heatmap_refiner_back_right = HeatmapMVF(**mvf_cfg)

        elif num_views == 2:
            self.heatmap_estimator_stereo_front = EgoPoseFormerHeatmap(encoder_cfg, num_heatmap, detach_heatmap_feat_init)

            self.heatmap_refiner_front_left = HeatmapMVF(**mvf_cfg)
            self.heatmap_refiner_front_right = HeatmapMVF(**mvf_cfg)

        elif num_views == 3:
            self.heatmap_estimator_stereo_front = EgoPoseFormerHeatmap(encoder_cfg, num_heatmap, detach_heatmap_feat_init)
            self.heatmap_estimator_stereo_back = EgoPoseFormerHeatmap(encoder_cfg, num_heatmap, detach_heatmap_feat_init)

            self.heatmap_refiner_front_left = HeatmapMVF(**mvf_cfg)
            self.heatmap_refiner_front_right = HeatmapMVF(**mvf_cfg)

            self.heatmap_refiner_back = HeatmapMVF(**mvf_cfg)

        input_dims = 128

        self.use_1by1_conv = self.heatmap_refiner_front_left.use_1by1_conv

        if self.use_1by1_conv:
            logger.info("")
            logger.info("We will use 1 by 1 conv in the original heatmap estimator")
            logger.info("")

        else:
            self.conv_heatmap_layers_stereo_front = nn.Sequential(
                nn.Conv2d(input_dims, input_dims, 1),
                nn.ReLU(inplace=False),
                nn.Conv2d(input_dims, input_dims * 2, 3, 2, 1),
                nn.ReLU(inplace=False),
                nn.Conv2d(input_dims * 2, input_dims * 2, 1),
                nn.ReLU(inplace=False),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(input_dims * 2, input_dims, 1),
                nn.ReLU(inplace=False),
                nn.Conv2d(input_dims, num_heatmap, 1)
            )

            if (self.num_views == 4) or (self.num_views == 3):
                self.conv_heatmap_layers_stereo_back = nn.Sequential(
                    nn.Conv2d(input_dims, input_dims, 1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(input_dims, input_dims * 2, 3, 2, 1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(input_dims * 2, input_dims * 2, 1),
                    nn.ReLU(inplace=False),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(input_dims * 2, input_dims, 1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(input_dims, num_heatmap, 1)
                )

    def get_anchors_2d_from_hm(self, heatmap):

        with torch.no_grad():
            B, V, C, H, W = heatmap.shape

            heatmap = heatmap.view(B * V, C, H, W)

            pts2d, maxvals, mask_valid = get_max_preds(
                heatmap, threshold=self.heatmap_threshold, normalize=True
            )

            pts2d = pts2d.view(B, V, self.num_heatmap, 2)
            maxvals = maxvals.view(B, V, self.num_heatmap)
            mask_valid = mask_valid.view(B, V, self.num_heatmap)

        return pts2d, maxvals, mask_valid

    def forward_heatmap_estimation_full(self, img):
        if (self.num_views == 4) or (self.num_views == 3):
            pred_heatmap_stereo_front, frame_feat_stereo_front, backbone_feat_stereo_front = self.heatmap_estimator_stereo_front(img[:, 0:2, :, :, :], return_feat=True)
            pred_heatmap_stereo_back, frame_feat_stereo_back, backbone_feat_stereo_back = self.heatmap_estimator_stereo_back(img[:, 2:, :, :, :], return_feat=True)

            pred_heatmap = torch.cat(
                (
                    pred_heatmap_stereo_front,
                    pred_heatmap_stereo_back,
                ),
                dim=1
            )

            frame_feat_multi_view = torch.cat((frame_feat_stereo_front, frame_feat_stereo_back), dim=1)

            list_backbone_feat_multi_view = []
            for idx in range(len(backbone_feat_stereo_front)):
                list_backbone_feat_multi_view.append(
                    torch.cat(
                        (
                            backbone_feat_stereo_front[idx],
                            backbone_feat_stereo_back[idx],
                        ),
                        dim=1
                    )
                )

        elif self.num_views == 2:
            pred_heatmap, frame_feat_multi_view, list_backbone_feat_multi_view = self.heatmap_estimator_stereo_front(img, return_feat=True)

        return pred_heatmap, frame_feat_multi_view, list_backbone_feat_multi_view

    def forward_heatmap_estimation_from_feat(self, frame_feat):
        B, V, C, H, W = frame_feat.size()

        if (self.num_views == 4) or (self.num_views == 3):

            frame_feat_stereo_front = frame_feat[:, 0:2]
            frame_feat_stereo_back = frame_feat[:, 2:]

            frame_feat_stereo_front = rearrange(frame_feat_stereo_front, "b v c h w -> (b v) c h w")
            frame_feat_stereo_back = rearrange(frame_feat_stereo_back, "b v c h w -> (b v) c h w")

            pred_heatmap_stereo_front = self.conv_heatmap_layers_stereo_front(frame_feat_stereo_front)
            pred_heatmap_stereo_back = self.conv_heatmap_layers_stereo_back(frame_feat_stereo_back)

            pred_heatmap_stereo_front = pred_heatmap_stereo_front.view(B, 2, self.num_heatmap, H, W)
            pred_heatmap_stereo_back = pred_heatmap_stereo_back.view(B, self.num_views - 2, self.num_heatmap, H, W)

            pred_heatmap = torch.cat(
                (
                    pred_heatmap_stereo_front,
                    pred_heatmap_stereo_back,
                ),
                dim=1
            )

        elif self.num_views == 2:

            frame_feat = frame_feat.view(B * 2, C, H, W)

            pred_heatmap = self.conv_heatmap_layers_stereo_front(frame_feat)

            pred_heatmap = pred_heatmap.view(B, 2, self.num_heatmap, H, W)

        return pred_heatmap

    def forward_heatmap_feat_estimation(self, img):
        if (self.num_views == 4) or (self.num_views == 3):
            frame_feat_stereo_front, backbone_feat_stereo_front = self.heatmap_estimator_stereo_front.forward_backbone(img[:, 0:2, :, :, :], return_feat=True)
            frame_feat_stereo_back, backbone_feat_stereo_back = self.heatmap_estimator_stereo_back.forward_backbone(img[:, 2:, :, :, :], return_feat=True)

            frame_feat_multi_view = torch.cat((frame_feat_stereo_front, frame_feat_stereo_back), dim=1)

            list_backbone_feat_multi_view = []
            for idx in range(len(backbone_feat_stereo_front)):
                list_backbone_feat_multi_view.append(
                    torch.cat(
                        (
                            backbone_feat_stereo_front[idx],
                            backbone_feat_stereo_back[idx],
                        ),
                        dim=1
                    )
                )

        elif self.num_views == 2:
            frame_feat_multi_view, list_backbone_feat_multi_view = self.heatmap_estimator_stereo_front.forward_backbone(img, return_feat=True)

        return frame_feat_multi_view, list_backbone_feat_multi_view

    def forward(self, img, heatmap_for_anchor=None):
        B, V, C, H, W = img.shape

        if self.use_1by1_conv:
            if self.full_training:
                heatmap_pred_init, frame_feat_multi_view_init, list_backbone_feat_multi_view_init = self.forward_heatmap_estimation_full(img)
            else:
                with torch.no_grad():
                    heatmap_pred_init, frame_feat_multi_view_init, list_backbone_feat_multi_view_init = self.forward_heatmap_estimation_full(img)

            backbone_feat_bottom_multi_view_init = list_backbone_feat_multi_view_init[-1]

            if self.use_pred_heatmap_init:
                heatmap_pred = heatmap_pred_init.detach().clone()

                if self.no_detach_feat_init:
                    frame_feat_multi_view = frame_feat_multi_view_init
                    backbone_feat_bottom_multi_view = backbone_feat_bottom_multi_view_init
                else:
                    frame_feat_multi_view = frame_feat_multi_view_init.detach().clone()
                    backbone_feat_bottom_multi_view = backbone_feat_bottom_multi_view_init.detach().clone()
            else:
                heatmap_pred = heatmap_pred_init
                frame_feat_multi_view = frame_feat_multi_view_init
                backbone_feat_bottom_multi_view = backbone_feat_bottom_multi_view_init

        else:

            if self.full_training:
                frame_feat_multi_view_init, list_backbone_feat_multi_view_init = self.forward_heatmap_feat_estimation(img)
            else:
                with torch.no_grad():
                    frame_feat_multi_view_init, list_backbone_feat_multi_view_init = self.forward_heatmap_feat_estimation(img)

            backbone_feat_bottom_multi_view_init = list_backbone_feat_multi_view_init[-1]

            if self.use_pred_heatmap_init:
                heatmap_pred_init = self.forward_heatmap_estimation_from_feat(frame_feat_multi_view_init.detach())

                heatmap_pred = heatmap_pred_init.detach().clone()

                if self.no_detach_feat_init:
                    frame_feat_multi_view = frame_feat_multi_view_init
                    backbone_feat_bottom_multi_view = backbone_feat_bottom_multi_view_init
                else:
                    frame_feat_multi_view = frame_feat_multi_view_init.detach().clone()
                    backbone_feat_bottom_multi_view = backbone_feat_bottom_multi_view_init.detach().clone()
            else:
                heatmap_pred_init = self.forward_heatmap_estimation_from_feat(frame_feat_multi_view_init)

                heatmap_pred = heatmap_pred_init
                frame_feat_multi_view = frame_feat_multi_view_init
                backbone_feat_bottom_multi_view = backbone_feat_bottom_multi_view_init

        list_heatmap_pred = [heatmap_pred_init]
        list_frame_feat = [frame_feat_multi_view_init]

        if isinstance(heatmap_for_anchor, torch.Tensor):
            anchors_2d, maxvals, anchors_valid = self.get_anchors_2d_from_hm(heatmap_for_anchor)
        else:
            anchors_2d, maxvals, anchors_valid = self.get_anchors_2d_from_hm(heatmap_pred_init)
        anchors_2d = anchors_2d.detach()

        if self.num_views == 4:

            heatmap_pred_front_left = heatmap_pred[:, 0, :, :, :]
            heatmap_pred_front_right = heatmap_pred[:, 1, :, :, :]
            heatmap_pred_back_left = heatmap_pred[:, 2, :, :, :]
            heatmap_pred_back_right = heatmap_pred[:, 3, :, :, :]

            frame_feat_front_left = frame_feat_multi_view[:, 0, :, :, :]
            frame_feat_front_right = frame_feat_multi_view[:, 1, :, :, :]
            frame_feat_back_left = frame_feat_multi_view[:, 2, :, :, :]
            frame_feat_back_right = frame_feat_multi_view[:, 3, :, :, :]

            backbone_feat_bottom_front_left = backbone_feat_bottom_multi_view[:, 0]
            backbone_feat_bottom_front_right = backbone_feat_bottom_multi_view[:, 1]
            backbone_feat_bottom_back_left = backbone_feat_bottom_multi_view[:, 2]
            backbone_feat_bottom_back_right = backbone_feat_bottom_multi_view[:, 3]

            list_heatmap_pred_refined_front_left, list_frame_feat_refined_front_left = self.heatmap_refiner_front_left(
                heatmap_pred_front_left, frame_feat_front_left, frame_feat_multi_view, anchors_2d, anchors_valid, backbone_feat_bottom_front_left, backbone_feat_bottom_multi_view
            )
            list_heatmap_pred_refined_front_right, list_frame_feat_refined_front_right = self.heatmap_refiner_front_right(
                heatmap_pred_front_right, frame_feat_front_right, frame_feat_multi_view, anchors_2d, anchors_valid, backbone_feat_bottom_front_right, backbone_feat_bottom_multi_view
            )
            list_heatmap_pred_refined_back_left, list_frame_feat_refined_back_left = self.heatmap_refiner_back_left(
                heatmap_pred_back_left, frame_feat_back_left, frame_feat_multi_view, anchors_2d, anchors_valid, backbone_feat_bottom_back_left, backbone_feat_bottom_multi_view
            )
            list_heatmap_pred_refined_back_right, list_frame_feat_refined_back_right = self.heatmap_refiner_back_right(
                heatmap_pred_back_right, frame_feat_back_right, frame_feat_multi_view, anchors_2d, anchors_valid, backbone_feat_bottom_back_right, backbone_feat_bottom_multi_view
            )

            for idx in range(len(list_heatmap_pred_refined_front_left)):
                list_heatmap_pred.append(
                    torch.stack(
                        (
                            list_heatmap_pred_refined_front_left[idx],
                            list_heatmap_pred_refined_front_right[idx],
                            list_heatmap_pred_refined_back_left[idx],
                            list_heatmap_pred_refined_back_right[idx]
                        ),
                        dim=1
                    )
                )

                list_frame_feat.append(
                    torch.stack(
                        (
                            list_frame_feat_refined_front_left[idx],
                            list_frame_feat_refined_front_right[idx],
                            list_frame_feat_refined_back_left[idx],
                            list_frame_feat_refined_back_right[idx]
                        ),
                        dim=1
                    )
                )

        elif self.num_views == 3:
            heatmap_pred_front_left = heatmap_pred[:, 0, :, :, :]
            heatmap_pred_front_right = heatmap_pred[:, 1, :, :, :]
            heatmap_pred_back = heatmap_pred[:, 2, :, :, :]

            frame_feat_front_left = frame_feat_multi_view[:, 0, :, :, :]
            frame_feat_front_right = frame_feat_multi_view[:, 1, :, :, :]
            frame_feat_back = frame_feat_multi_view[:, 2, :, :, :]

            backbone_feat_bottom_front_left = backbone_feat_bottom_multi_view[:, 0]
            backbone_feat_bottom_front_right = backbone_feat_bottom_multi_view[:, 1]
            backbone_feat_bottom_back = backbone_feat_bottom_multi_view[:, 2]

            list_heatmap_pred_refined_front_left, list_frame_feat_refined_front_left = self.heatmap_refiner_front_left(
                heatmap_pred_front_left, frame_feat_front_left, frame_feat_multi_view, anchors_2d, anchors_valid, backbone_feat_bottom_front_left, backbone_feat_bottom_multi_view
            )
            list_heatmap_pred_refined_front_right, list_frame_feat_refined_front_right = self.heatmap_refiner_front_right(
                heatmap_pred_front_right, frame_feat_front_right, frame_feat_multi_view, anchors_2d, anchors_valid, backbone_feat_bottom_front_right, backbone_feat_bottom_multi_view
            )
            list_heatmap_pred_refined_back, list_frame_feat_refined_back = self.heatmap_refiner_back(
                heatmap_pred_back, frame_feat_back, frame_feat_multi_view, anchors_2d, anchors_valid, backbone_feat_bottom_back, backbone_feat_bottom_multi_view
            )

            for idx in range(len(list_heatmap_pred_refined_front_left)):
                list_heatmap_pred.append(
                    torch.stack(
                        (
                            list_heatmap_pred_refined_front_left[idx],
                            list_heatmap_pred_refined_front_right[idx],
                            list_heatmap_pred_refined_back[idx],
                        ),
                        dim=1
                    )
                )

                list_frame_feat.append(
                    torch.stack(
                        (
                            list_frame_feat_refined_front_left[idx],
                            list_frame_feat_refined_front_right[idx],
                            list_frame_feat_refined_back[idx],
                        ),
                        dim=1
                    )
                )

        elif self.num_views == 2:
            heatmap_pred_front_left = heatmap_pred[:, 0, :, :, :]
            heatmap_pred_front_right = heatmap_pred[:, 1, :, :, :]
            frame_feat_front_left = frame_feat_multi_view[:, 0, :, :, :]
            frame_feat_front_right = frame_feat_multi_view[:, 1, :, :, :]

            backbone_feat_bottom_front_left = backbone_feat_bottom_multi_view[:, 0]
            backbone_feat_bottom_front_right = backbone_feat_bottom_multi_view[:, 1]

            list_heatmap_pred_refined_front_left, list_frame_feat_refined_front_left = self.heatmap_refiner_front_left(
                heatmap_pred_front_left, frame_feat_front_left, frame_feat_multi_view, anchors_2d, anchors_valid, backbone_feat_bottom_front_left, backbone_feat_bottom_multi_view
            )
            list_heatmap_pred_refined_front_right, list_frame_feat_refined_front_right = self.heatmap_refiner_front_right(
                heatmap_pred_front_right, frame_feat_front_right, frame_feat_multi_view, anchors_2d, anchors_valid, backbone_feat_bottom_front_right, backbone_feat_bottom_multi_view
            )

            for idx in range(len(list_heatmap_pred_refined_front_left)):
                list_heatmap_pred.append(
                    torch.stack(
                        (
                            list_heatmap_pred_refined_front_left[idx],
                            list_heatmap_pred_refined_front_right[idx],
                        ),
                        dim=1
                    )
                )

                list_frame_feat.append(
                    torch.stack(
                        (
                            list_frame_feat_refined_front_left[idx],
                            list_frame_feat_refined_front_right[idx],
                        ),
                        dim=1
                    )
                )

        return list_heatmap_pred, list_frame_feat




class HeatmapMVF(nn.Module):
    def __init__(
        self,
        input_dims,
        embed_dims,
        num_former_layers,
        image_size,
        feat_down_stride,
        detach_heatmap_feat,
        mvf_transformer_cfg,
        heatmap_threshold,
        num_views,
        num_heatmap,
        joint_query_adaptation=False,
        joint_query_adaptation_multi_view=False,
        joint_query_only=False,
        use_1by1_conv=False,
    ):
        super(HeatmapMVF, self).__init__()

        self.num_heatmap = num_heatmap
        self.num_views = num_views
        self.heatmap_threshold = heatmap_threshold
        self.detach_heatmap_feat = detach_heatmap_feat
        self.joint_query_adaptation = joint_query_adaptation
        self.joint_query_adaptation_multi_view = joint_query_adaptation_multi_view
        self.joint_query_only = joint_query_only
        self.use_1by1_conv = use_1by1_conv

        self.feat_shape = (
            image_size[0] // feat_down_stride,
            image_size[1] // feat_down_stride,
        )

        if joint_query_adaptation:
            assert joint_query_adaptation_multi_view == False

            self.heatmap_proj = nn.Sequential(
                nn.Linear(self.feat_shape[0]*self.feat_shape[1], embed_dims),
                nn.ReLU(inplace=False),
                nn.Linear(embed_dims, embed_dims),
            )

            self.fc_bfb = nn.Linear(512, embed_dims)
            self.fc_query = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(inplace=False),
                )
            self.joint_query_embed = nn.Embedding(num_heatmap, embed_dims)

        elif joint_query_adaptation_multi_view:
            assert joint_query_adaptation == False

            self.heatmap_proj = nn.Sequential(
                nn.Linear(self.feat_shape[0]*self.feat_shape[1], embed_dims),
                nn.ReLU(inplace=False),
                nn.Linear(embed_dims, embed_dims),
            )

            self.fc_bfb = nn.Linear(512 * num_views, embed_dims)
            self.fc_query = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(inplace=False),
                )
            self.joint_query_embed = nn.Embedding(num_heatmap, embed_dims)

        elif joint_query_only:
            self.joint_query_embed = nn.Embedding(num_heatmap, embed_dims)
            self.fc_query = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(inplace=False),
                )
        else:
            self.heatmap_proj = nn.Sequential(
                nn.Linear(self.feat_shape[0]*self.feat_shape[1], embed_dims),
                nn.ReLU(inplace=False),
                nn.Linear(embed_dims, embed_dims),
            )
            self.query_pos_embed = nn.Parameter(torch.zeros(1, num_heatmap, embed_dims))

        self.frame_feat_multi_view_proj = nn.Conv2d(input_dims, embed_dims, 1, 1, 0)
        self.frame_feat_multi_view_pos_embed = nn.Parameter(torch.zeros(1, num_views, self.feat_shape[0]*self.feat_shape[1], embed_dims))

        self.frame_feat_proj_layers = nn.Sequential(
            nn.Conv2d(input_dims, input_dims * 2, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(input_dims * 2, input_dims * 4, 3, 2, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(input_dims * 4, input_dims, 1),
            nn.ReLU(inplace=False),
        )

        self.transformer_layers = nn.ModuleList()
        for idx in range(num_former_layers):
            _mvf_transformer_cfg = copy.deepcopy(mvf_transformer_cfg)
            _mvf_transformer_cfg.update({
                "num_views": num_views,
                "embed_dims": embed_dims,
                "feat_shape": self.feat_shape,
            })
            self.transformer_layers.append(MultiViewTransformerLayer(**_mvf_transformer_cfg))

        self.post_norm = torch.nn.ModuleList(
            [nn.LayerNorm(embed_dims) for _ in range(num_former_layers)]
        )

        self.head_layers = nn.ModuleList()
        for idx in range(num_former_layers):
            self.head_layers.append(TransformerHeadLayer(input_dims=num_heatmap, output_dims=input_dims))


        self.frame_feat_refined_proj_layers = nn.ModuleList()
        for idx in range(num_former_layers):
            self.frame_feat_refined_proj_layers.append(
                nn.Sequential(
                    nn.Conv2d(input_dims, input_dims, 1),
                    nn.ReLU(inplace=False),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(input_dims, input_dims, 1),
                    nn.ReLU(inplace=False),
                )
            )

        if self.use_1by1_conv:
            self.conv_heatmap = nn.Conv2d(input_dims, num_heatmap, 1)
            logger.info("")
            logger.info("We will use 1 by 1 conv in the MVF module")
            logger.info("")
        else:
            self.conv_heatmap_layers = nn.ModuleList()
            for idx in range(num_former_layers):
                self.conv_heatmap_layers.append(
                    nn.Sequential(
                        nn.Conv2d(input_dims, input_dims * 2, 3, 2, 1),
                        nn.ReLU(inplace=False),
                        nn.Conv2d(input_dims * 2, input_dims * 2, 1),
                        nn.ReLU(inplace=False),
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                        nn.Conv2d(input_dims * 2, input_dims, 1),
                        nn.ReLU(inplace=False),
                        nn.Conv2d(input_dims, num_heatmap, 1)
                    )
                )

    def forward_feat_only(self, heatmap, frame_feat, frame_feat_multi_view, anchors_2d, anchors_valid, backbone_feat_bottom, backbone_feat_bottom_multi_view):
        B, V, C, H, W = frame_feat_multi_view.size()

        if self.joint_query_adaptation:
            heatmap = rearrange(heatmap, "b c h w -> b c (h w)")
            heatmap_embed = self.heatmap_proj(heatmap)

            bfb = F.adaptive_avg_pool2d(backbone_feat_bottom, (1, 1))
            bfb = bfb.view(B, -1)
            bfb = self.fc_bfb(bfb)
            bfb = bfb.unsqueeze(1)

            joint_query_embed = self.joint_query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
            x = self.fc_query(joint_query_embed + bfb + heatmap_embed)

        elif self.joint_query_adaptation_multi_view:
            heatmap = rearrange(heatmap, "b c h w -> b c (h w)")
            heatmap_embed = self.heatmap_proj(heatmap)

            bfb = F.adaptive_avg_pool2d(backbone_feat_bottom_multi_view, (1, 1))
            bfb = bfb.view(B, -1)
            bfb = self.fc_bfb(bfb)
            bfb = bfb.unsqueeze(1)

            joint_query_embed = self.joint_query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
            x = self.fc_query(joint_query_embed + bfb + heatmap_embed)

        elif self.joint_query_only:
            joint_query_embed = self.joint_query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
            x = self.fc_query(joint_query_embed)

        else:
            heatmap = rearrange(heatmap, "b c h w -> b c (h w)")
            heatmap_embed = self.heatmap_proj(heatmap)

            x = heatmap_embed + self.query_pos_embed

        frame_feat_multi_view = rearrange(frame_feat_multi_view, "b v c h w -> (b v) c h w")
        frame_feat_multi_view = self.frame_feat_multi_view_proj(frame_feat_multi_view)
        frame_feat_multi_view = rearrange(frame_feat_multi_view, "(b v) c h w -> b v (h w) c", b=B)
        frame_feat_multi_view = frame_feat_multi_view + self.frame_feat_multi_view_pos_embed

        frame_feat = self.frame_feat_proj_layers(frame_feat)

        list_frame_feat_refined = []
        for idx in range(len(self.transformer_layers)):
            x = self.transformer_layers[idx](
                x,
                frame_feat_multi_view,
                anchors_2d,
                anchors_valid,
            )

            _x = self.post_norm[idx](x)

            _, _, HW = _x.size()
            h = w = int(math.sqrt(HW))
            _x = rearrange(_x, "b j (h w) -> b j h w", b=B, h=h, w=w)

            offset_pred = self.head_layers[idx](_x)
            frame_feat_refined = self.frame_feat_refined_proj_layers[idx](offset_pred + frame_feat.detach())

            list_frame_feat_refined.append(frame_feat_refined)

        return list_frame_feat_refined

    def forward(self, heatmap, frame_feat, frame_feat_multi_view, anchors_2d, anchors_valid, backbone_feat_bottom, backbone_feat_bottom_multi_view):
        B, V, C, H, W = frame_feat_multi_view.size()

        if self.joint_query_adaptation:
            heatmap = rearrange(heatmap, "b c h w -> b c (h w)")
            heatmap_embed = self.heatmap_proj(heatmap)

            bfb = F.adaptive_avg_pool2d(backbone_feat_bottom, (1, 1))
            bfb = bfb.view(B, -1)
            bfb = self.fc_bfb(bfb)
            bfb = bfb.unsqueeze(1)

            joint_query_embed = self.joint_query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
            x = self.fc_query(joint_query_embed + bfb + heatmap_embed)

        elif self.joint_query_adaptation_multi_view:
            heatmap = rearrange(heatmap, "b c h w -> b c (h w)")
            heatmap_embed = self.heatmap_proj(heatmap)

            bfb = F.adaptive_avg_pool2d(backbone_feat_bottom_multi_view, (1, 1))
            bfb = bfb.view(B, -1)
            bfb = self.fc_bfb(bfb)
            bfb = bfb.unsqueeze(1)

            joint_query_embed = self.joint_query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
            x = self.fc_query(joint_query_embed + bfb + heatmap_embed)

        elif self.joint_query_only:
            joint_query_embed = self.joint_query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
            x = self.fc_query(joint_query_embed)

        else:
            heatmap = rearrange(heatmap, "b c h w -> b c (h w)")
            heatmap_embed = self.heatmap_proj(heatmap)

            x = heatmap_embed + self.query_pos_embed

        # memory
        frame_feat_multi_view = rearrange(frame_feat_multi_view, "b v c h w -> (b v) c h w")
        frame_feat_multi_view = self.frame_feat_multi_view_proj(frame_feat_multi_view)
        frame_feat_multi_view = rearrange(frame_feat_multi_view, "(b v) c h w -> b v (h w) c", b=B)
        frame_feat_multi_view = frame_feat_multi_view + self.frame_feat_multi_view_pos_embed

        frame_feat = self.frame_feat_proj_layers(frame_feat)

        list_frame_feat_refined = []
        list_heatmap_refined = []
        for idx in range(len(self.transformer_layers)):
            x = self.transformer_layers[idx](
                x,
                frame_feat_multi_view,
                anchors_2d,
                anchors_valid,
            )

            _x = self.post_norm[idx](x)

            _, _, HW = _x.size()
            h = w = int(math.sqrt(HW))
            _x = rearrange(_x, "b j (h w) -> b j h w", b=B, h=h, w=w)

            offset_pred = self.head_layers[idx](_x)

            frame_feat_refined = self.frame_feat_refined_proj_layers[idx](offset_pred + frame_feat.detach())

            if self.detach_heatmap_feat:
                if self.use_1by1_conv:
                    heatmap_refined = self.conv_heatmap(frame_feat_refined.detach())
                else:
                    heatmap_refined = self.conv_heatmap_layers[idx](frame_feat_refined.detach())
            else:
                if self.use_1by1_conv:
                    heatmap_refined = self.conv_heatmap(frame_feat_refined)
                else:
                    heatmap_refined = self.conv_heatmap_layers[idx](frame_feat_refined)

            list_frame_feat_refined.append(frame_feat_refined)
            list_heatmap_refined.append(heatmap_refined)

        return list_heatmap_refined, list_frame_feat_refined


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""
    def __init__(self, img_size=[64, 64], patch_size=[4, 4], in_chans=128, embed_dim=1024,
                norm_layer=None, flatten=True, bias=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)

        x = self.norm(x)
        return x, H, W


class DeformMultiViewAttn(MSDeformAttn):
    def __init__(self, feat_shape, **kwargs):
        _init_cfg = {
            "d_model": kwargs.pop("embed_dim"),
            "n_heads": kwargs.pop("num_heads"),
            "n_points": 16
        }
        super(DeformMultiViewAttn, self).__init__(**_init_cfg)

        self.register_buffer("spatial_shapes", torch.tensor([[feat_shape[0], feat_shape[1]]], dtype=torch.long))
        self.register_buffer("start_index", torch.tensor([0,], dtype=torch.long))

    def forward(self, query, img_feat, anchors_2d):
        B, J, C = query.shape

        anchors_2d = anchors_2d.detach()

        _q = query.reshape(B, J, C)
        _kv = img_feat.reshape(B, -1, C)
        _ref_pts = anchors_2d.reshape(B, J, 1, 2)

        out = super(DeformMultiViewAttn, self).forward(
            _q,
            _ref_pts,
            _kv,
            self.spatial_shapes,
            self.start_index
        )
        out = out.reshape(B, J, C)
        return out


class SpatialMHA(CustomMultiheadAttention):
    def forward(self, q, k, v, bias):

        B, J, C = q.shape

        _q = self.q_proj(q).reshape(B, J, self.num_heads, self.head_dims).permute(0, 2, 1, 3)  # [B, H, J, c]
        _k = self.k_proj(k).reshape(B, J, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        _v = self.v_proj(v).reshape(B, J, self.num_heads, self.head_dims).permute(0, 2, 1, 3)

        attn = (_q @ _k.transpose(-2, -1)) * self.scale  # [B, H, J, J]
        if bias is not None:
            attn = attn + bias

        attn = attn.softmax(dim=-1)

        x = (attn @ _v).permute(0, 2, 1, 3).reshape(B, J, C)
        if self.out_proj is not None:
            x = self.out_proj(x)
        return x


class MultiViewTransformerLayer(nn.Module):
    def __init__(
        self,
        num_views,
        embed_dims,
        cross_attn_cfg,
        spatial_attn_cfg,
        ffn_cfg,
        feat_shape,
        use_normal_cross_attn=False,
    ):
        super().__init__()
        self.embed_dims = embed_dims

        _cross_attn_cfg = copy.deepcopy(cross_attn_cfg)
        _cross_attn_cfg.update({
            "embed_dim": embed_dims,
            "feat_shape": feat_shape
        })

        self.use_normal_cross_attn = use_normal_cross_attn
        if use_normal_cross_attn:
            self.cross_attn = nn.MultiheadAttention(embed_dims, 4)

        else:
            self.cross_attn = DeformMultiViewAttn(**_cross_attn_cfg)


        self.fuse_mlp = nn.Linear(num_views * embed_dims, embed_dims)
        self.norm_cross = nn.LayerNorm(embed_dims)

        _spatial_attn_cfg = copy.deepcopy(spatial_attn_cfg)
        _spatial_attn_cfg.update({"embed_dim": embed_dims})
        self.spatial_attn = SpatialMHA(**_spatial_attn_cfg)
        self.norm_spatial = nn.LayerNorm(embed_dims)

        _ffn_cfg = copy.deepcopy(ffn_cfg)
        _ffn_cfg.update({"embed_dims": embed_dims})
        self.ffn = FFN(**_ffn_cfg)
        self.norm_ffn = nn.LayerNorm(embed_dims)

    def _run_spatial_attn(self, feat_query):
        identity = feat_query

        q = feat_query
        k = q
        v = feat_query

        attn_res = self.spatial_attn(q, k, v, bias=None)
        x = identity + attn_res
        x = self.norm_spatial(x)
        return x


    def _run_cross_attn(
        self,
        feat_query,
        frame_feat_multi_view,
        anchors_2d,
        anchors_valid,
    ):
        B, V, J = frame_feat_multi_view.shape[:3]
        identity = feat_query

        q = feat_query
        feats_per_view = []

        if self.use_normal_cross_attn:
            for i in range(V):
                qi = q
                feat_i = frame_feat_multi_view[:, i]
                anchors_i = anchors_2d[:, i]

                qi = rearrange(qi, "b p c -> p b c")
                feat_i = rearrange(feat_i, "b p c -> p b c")

                attn_res = self.cross_attn(query=qi,
                                        key=feat_i,
                                        value=feat_i,
                                        )[0]

                attn_res = rearrange(attn_res, "p b c -> b p c")
                feats_per_view.append(attn_res)

        else:
            for i in range(V):
                qi = q
                feat_i = frame_feat_multi_view[:, i]
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
        feat_query,
        frame_feat_multi_view,
        anchors_2d,
        anchors_valid,
    ):
        x = feat_query
        x = self._run_cross_attn(x, frame_feat_multi_view, anchors_2d, anchors_valid)
        x = self._run_spatial_attn(x)
        x = self._forward_ffn(x)
        return x


class TransformerHeadLayer(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims
    ):
        super().__init__()


        if output_dims == 128:
            self.head = nn.Sequential(
                nn.Conv2d(input_dims, int(output_dims // 2), 1),
                nn.ReLU(inplace=False),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(int(output_dims // 2), output_dims, 1),
                nn.ReLU(inplace=False),
            )
        elif output_dims == 512:
            self.head = nn.Sequential(
                nn.Conv2d(input_dims, input_dims, 1),
                nn.ReLU(inplace=False),
                nn.Conv2d(input_dims, int(output_dims // 8), 1),
                nn.ReLU(inplace=False),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(int(output_dims // 8), int(output_dims // 4), 1),
                nn.ReLU(inplace=False),
                nn.Conv2d(int(output_dims // 4), output_dims, 1),
                nn.ReLU(inplace=False),
            )

    def forward(self, x):
        x = self.head(x)
        return x

