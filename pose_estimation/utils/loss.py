# Author: Hiroyasu Akada

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_mpjpe_batch(pred_keypoints, gt_keypoints):
    error = torch.linalg.norm(pred_keypoints - gt_keypoints, dim=-1, ord=2)
    mpjpe = error.mean(dim=1)
    return mpjpe


def compute_pck_3d(pred_keypoints, gt_keypoints, threshold=150):
    num_keypoints = gt_keypoints.size()[0]
    distances = torch.linalg.norm(gt_keypoints - pred_keypoints, axis=1)
    num_correct = torch.sum(distances <= threshold)
    pck = num_correct / num_keypoints
    return pck


def compute_auc_3d(pred_keypoints, gt_keypoints, thresholds=np.linspace(0, 150, 31).tolist()):
    pck_scores = []
    for threshold in thresholds:
        pck = compute_pck_3d(pred_keypoints, gt_keypoints, threshold)
        pck_scores.append(pck)

    return torch.mean(torch.tensor(pck_scores))


def compute_pck_3d_batch(pred_keypoints, gt_keypoints, threshold=150):
    distances = torch.linalg.norm(gt_keypoints - pred_keypoints, axis=-1)
    num_correct = torch.sum(distances <= threshold, axis=1)
    num_keypoints = gt_keypoints.size()[1]
    pck = num_correct / num_keypoints
    return pck


def compute_auc_3d_batch(pred_keypoints, gt_keypoints, thresholds=np.linspace(0, 150, 31).tolist()):
    pck_scores = []
    for threshold in thresholds:
        pck = compute_pck_3d_batch(pred_keypoints, gt_keypoints, threshold)
        pck_scores.append(pck)

    pck_scores = torch.stack(pck_scores, axis=-1)
    auc_scores = torch.mean(pck_scores, axis=-1)
    return auc_scores


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)

def get_max_preds(heatmaps, threshold=0.5, normalize=False):
    assert isinstance(heatmaps, torch.Tensor), 'heatmaps should be a torch.Tensor'
    assert heatmaps.ndim == 4, 'heatmaps should be 4-ndim'

    B, J, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.view(B, J, -1)

    maxvals, idx = torch.max(heatmaps_reshaped, dim=2, keepdim=True)

    preds = torch.tile(idx, (1, 1, 2)).float()

    preds[:, :, 0] = preds[:, :, 0] % W  # x-coordinate
    preds[:, :, 1] = preds[:, :, 1] // W  # y-coordinate

    if normalize:
        preds[:, :, 0] = preds[:, :, 0] / W  # x-coordinate
        preds[:, :, 1] = preds[:, :, 1] / H  # y-coordinate

    valid_mask = maxvals >= threshold

    return preds, maxvals.squeeze(), valid_mask.squeeze()


def get_max_preds_soft_pytorch(batch_heatmaps, normalize=False):
    assert len(batch_heatmaps.shape) == 4, 'batch_images should be 4-ndim (B, J, H, W)'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    height = batch_heatmaps.shape[2]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.view((batch_size, num_joints, -1))

    maxvals = heatmaps_reshaped.max(dim=2)[0]
    maxvals = maxvals.view((batch_size, num_joints, 1))

    heatmaps_reshaped = F.softmax(heatmaps_reshaped, dim=2)
    batch_heatmaps = heatmaps_reshaped.view(batch_size, num_joints, height, width)

    x = batch_heatmaps.sum(dim = 2)
    y = batch_heatmaps.sum(dim = 3)
    x_indices = torch.cuda.comm.broadcast(torch.arange(width).type(torch.cuda.FloatTensor), devices=[x.device.index])[0]
    y_indices = torch.cuda.comm.broadcast(torch.arange(height).type(torch.cuda.FloatTensor), devices=[x.device.index])[0]
    x_indices = x_indices.view(1,1,width)
    y_indices = y_indices.view(1,1,height)
    x *= x_indices
    y *= y_indices
    x = x.sum(dim = 2, keepdim=True)
    y = y.sum(dim = 2, keepdim=True)

    if normalize:
        x = x / width
        y = y / height

    preds = torch.cat([x, y], dim=2)

    return preds, maxvals


class JointsCoordinateLoss(nn.Module):
    def __init__(self, use_target_weight, loss_type='sl1', image_size=(384, 288)):
        super(JointsCoordinateLoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.loss_type = loss_type
        self.image_size = image_size
        return

    def forward(self, output, target, target_weight):
        preds, _ = get_max_preds_soft_pytorch(output)
        preds[:, :, 0] /= self.image_size[1]
        preds[:, :, 1] /= self.image_size[0]
        target[:, :, 0] /= self.image_size[1]
        target[:, :, 1] /= self.image_size[0]
        if self.loss_type == 'sl1':
            loss = F.smooth_l1_loss(preds, target)
        elif self.loss_type == 'wing':
            raise NotImplementedError
        else:
            raise NotImplementedError
        return loss

class WingLoss(nn.Module):
    def __init__(self, use_target_weight, width=5, curvature=0.5, image_size=(384, 288)):
        super(WingLoss, self).__init__()
        self.width = width
        self.curvature = curvature
        self.C = self.width - self.width * np.log(1 + self.width / self.curvature)
        self.image_size = image_size

    def forward(self, output, target, target_weight):
        prediction, _ = get_max_preds_soft_pytorch(output)
        prediction[:, :, 0] /= self.image_size[1]
        prediction[:, :, 1] /= self.image_size[0]
        target[:, :, 0] /= self.image_size[1]
        target[:, :, 1] /= self.image_size[0]
        diff = target - prediction
        diff_abs = diff.abs()
        loss = diff_abs.clone()

        idx_smaller = diff_abs < self.width
        idx_bigger = diff_abs >= self.width

        loss[idx_smaller] = self.width * torch.log(1 + diff_abs[idx_smaller] / self.curvature)
        loss[idx_bigger]  = loss[idx_bigger] - self.C
        loss = loss.mean()
        return loss