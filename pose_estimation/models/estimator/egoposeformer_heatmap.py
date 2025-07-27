import torch.nn as nn

from pose_estimation.models.backbones.resnet import ResnetBackbone

from einops import rearrange
from torch._dynamo import allow_in_graph
allow_in_graph(rearrange)

class EgoPoseFormerHeatmap(nn.Module):
    def __init__(
        self,
        encoder_cfg,
        num_heatmap,
        detach_heatmap_feat_init=False,
        **kwargs
    ):
        super(EgoPoseFormerHeatmap, self).__init__()

        self.num_heatmap = num_heatmap
        self.detach_heatmap_feat_init = detach_heatmap_feat_init

        self.encoder = ResnetBackbone(**encoder_cfg)
        self.conv_heatmap = nn.Conv2d(self.encoder.get_output_channel(), num_heatmap, 1)

    def forward_backbone(self, img, return_feat=False):
        feats, backbone_feats = self.encoder(img)
        return feats, backbone_feats

    def forward(self, img, return_feat=False):
        B, V, img_c, img_h, img_w = img.shape

        feats, backbone_feats = self.forward_backbone(img)

        if self.detach_heatmap_feat_init:
            heatmap_pred = self.conv_heatmap(feats.view(B * V, *feats.shape[2:]).detach())
        else:
            heatmap_pred = self.conv_heatmap(feats.view(B * V, *feats.shape[2:]))

        heatmap_pred = heatmap_pred.view(B, V, *heatmap_pred.shape[1:])

        if return_feat:
            return heatmap_pred, feats, backbone_feats
        else:
            return heatmap_pred
