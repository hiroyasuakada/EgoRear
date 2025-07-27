# Author: Hiroyasu Akada

import os
from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed

from pytorch_lightning import LightningModule
from pytorch_lightning.strategies import ParallelStrategy

from torch.optim.lr_scheduler import MultiStepLR

from pose_estimation.models.estimator import EgoPoseFormerMVFEX
from pose_estimation.datasets.dataset import get_dataset

from pose_estimation.models.utils.pose_metric import (
    MpjpeLoss,
    batch_compute_similarity_transform_numpy
)

from pose_estimation.utils.state_dict import fix_model_state_dict
from pose_estimation.utils.loss import compute_pck_3d_batch, compute_auc_3d_batch, compute_mpjpe_batch

from loguru import logger




class Pose3DMVFEXLightningModel(LightningModule):
    def __init__(
        self,
        model_cfg: dict,
        dataset_type: str,
        data_root: str,
        pose_relative_type: str,
        lr: float,
        encoder_lr_scale: float,
        weight_decay: float,
        lr_decay_epochs: tuple,
        warmup_iters: int,
        w_mpjpe: float,
        w_heatmap: float,
        batch_size: int,
        workers: int,
        compile: bool,
        compile_mode: str,
        save_result: bool,
        network_pretrained: str = None,
        heatmap_estimator_mvf_pretrained=None,
        test_on_rw=False,
        dataset_kwargs: dict = {}
    ):
        super().__init__()

        self.test_on_rw = test_on_rw
        if test_on_rw:
            model_cfg["pose3d_cfg"]["camera_model"] = "ego4view_rw"
            dataset_type = "ego4view_rw_pose3d"
            data_root = "/scratch/inf0/user/hakada/Ego4View_rw_impl"

        self.save_hyperparameters()

        self.dataset_type = dataset_type

        self.dataset_kwargs = dataset_kwargs
        self.pose_relative_type = pose_relative_type

        self.compile = compile
        self.compile_mode = compile_mode
        self.network = EgoPoseFormerMVFEX(**model_cfg)

        self.lr = lr
        self.encoder_lr_scale = encoder_lr_scale
        self.weight_decay = weight_decay
        self.lr_decay_epochs = lr_decay_epochs
        self.warmup_iters = warmup_iters
        self.w_mpjpe = w_mpjpe
        self.w_heatmap = w_heatmap

        self.data_root = data_root
        self.batch_size = batch_size
        self.workers = workers

        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None

        self.criteria = MpjpeLoss()

        self.cm2mm = 10

        self.log_img_every_n_batches = 2500

        self.save_result = save_result
        self.save_result_every_n_batches = 1

        if network_pretrained:
            self.load_trained_model(self.network, network_pretrained)

        if heatmap_estimator_mvf_pretrained:
            self.load_trained_model(self.network.heatmap_estimator, heatmap_estimator_mvf_pretrained)

    def load_trained_model(self, network, path_to_trained_weights):
        state_dict = torch.load(path_to_trained_weights, map_location="cpu", weights_only=False)["state_dict"]
        state_dict = fix_model_state_dict(state_dict, rm_name="network._orig_mod.")
        network.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        self.network.train()

        img = batch["img"]
        gt_heatmap = batch["gt_heatmap"]
        gt_pose = batch["gt_pose"]

        B, V, img_c, img_h, img_w = img.shape

        if (self.dataset_type == "ego4view_rw_pose3d"):
            coord_trans_mat = batch["coord_trans_mat"]
        else:
            coord_trans_mat = None


        origin_3d = None
        pred_pose_all, pred_heatmap_all = self.network(img, coord_trans_mat, origin_3d)

        loss_dict = OrderedDict()
        for i, pred_pose in enumerate(pred_pose_all):
            losses_i = self.get_pose_loss(pred_pose, gt_pose)
            for k, v in losses_i.items():
                loss_dict["%s_%d"%(k, i)] = v

        list_gt_heatmap = [gt_heatmap[:, l] for l in range(V)]
        for id, pred_heatmap in enumerate(pred_heatmap_all):
            list_pred_heatmap = [pred_heatmap[:, l] for l in range(V)]
            loss_id = sum([self.get_heatmap_loss(x, y) for x, y in zip(list_pred_heatmap, list_gt_heatmap)])
            loss_dict["heatmap_loss_{}".format(id)] = loss_id

        loss_total = sum(loss_dict.values())

        for k, v in loss_dict.items():
            self.log("train/{}".format(k), v)
        self.log("train/loss_total", loss_total)

        return loss_total

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        if (batch_idx != 0) and (batch_idx % self.save_result_every_n_batches == 0):
            self.eval_step(batch, batch_idx, "predict")

    def eval_step(self, batch, batch_idx, mode):
        self.network.eval()

        img = batch["img"]
        if (mode == "val") or (mode == "test"):
            gt_heatmap = batch["gt_heatmap"]
        gt_pose = batch["gt_pose"]

        B, V, img_c, img_h, img_w = img.shape

        if (self.dataset_type == "ego4view_rw_pose3d") or (self.dataset_type == "unrealego_rw_pose3d"):
            coord_trans_mat = batch["coord_trans_mat"]
        else:
            coord_trans_mat = None

        if (self.dataset_type == "unrealego") and (self.pose_relative_type == "pelvis"):
            origin_3d = batch["origin_3d"]
        else:
            origin_3d = None

        pred_pose_all, pred_heatmap_all = self.network(img, coord_trans_mat, origin_3d)

        pred_pose_proposal = pred_pose_all[0]
        pred_pose_final = pred_pose_all[-1]
        pred_pose_proposal_output = pred_pose_proposal.detach().cpu().numpy()
        pred_pose_final_output = pred_pose_final.detach().cpu().numpy()

        if (mode == "val") or (mode == "test"):
            output_dict = OrderedDict()

            output_dict["pred_pose_final"] = pred_pose_final_output
            output_dict["pred_pose_proposal"] = pred_pose_proposal_output

            if gt_pose is not None:
                metrics_final = self.evaluate_pose(pred_pose_final, gt_pose, "final")
                metrics_proposal = self.evaluate_pose(pred_pose_proposal, gt_pose, "proposal")

                output_dict.update(metrics_final)
                output_dict.update(metrics_proposal)

            for k, v in output_dict.items():
                if (mode == "val") and ("mpjpe" not in k):
                    continue
                self.log("{}/{}".format(mode, k), v.mean(), sync_dist=True)

        return None

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure,):
        optimizer.step(closure=optimizer_closure)
        if self.trainer.global_step < self.warmup_iters:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.warmup_iters))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

    def configure_optimizers(self):
        if self.encoder_lr_scale == 1.0:
            no_decay_params = []
            other_params = []
            for name, param in self.network.named_parameters():
                if 'norm' in name or 'bn' in name or 'ln' in name or 'bias' in name:
                    no_decay_params.append(param)
                else:
                    other_params.append(param)
            optimizer = optim.AdamW(
                [
                    {'params': no_decay_params, 'weight_decay': 0.0},
                    {'params': other_params, 'weight_decay': self.weight_decay}
                ],
                lr=self.lr
            )
        else:
            param_groups = [
                {'params': self.network.encoder.parameters(), 'lr': self.lr * self.encoder_lr_scale}
            ]
            param_groups.append(
                {
                    'params': [
                        param for name, param in self.network.named_parameters() if not name.startswith('encoder')
                    ]
                }
            )
            optimizer = optim.AdamW(param_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = MultiStepLR(optimizer, self.lr_decay_epochs, gamma=0.1)
        return [optimizer], [scheduler]

    def setup(self, stage: str):
        logger.info("Setting up [{}] dataset...".format(stage))

        if isinstance(self.trainer.strategy, ParallelStrategy):
            num_processes = max(1, self.trainer.strategy.num_processes)
            self.batch_size = int(self.batch_size / num_processes)
            self.workers = int(self.workers / num_processes)

        if stage == "fit":
            self.train_dataset = get_dataset(self.dataset_type, self.data_root, "train", **self.dataset_kwargs)
            logger.info('train data = {}'.format(len(self.train_dataset)))

        if stage == "test" or stage == "predict":
            self.eval_dataset = get_dataset(self.dataset_type, self.data_root, "test", **self.dataset_kwargs)
            logger.info('test data = {}'.format(len(self.eval_dataset)))

        else:
            self.eval_dataset = get_dataset(self.dataset_type, self.data_root, "validation", **self.dataset_kwargs)
            logger.info('validation data = {}'.format(len(self.eval_dataset)))

        logger.info("")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False
        )

    def get_pose_loss(self, pred_pose, gt_pose):
        mpjpe_loss = self.criteria(pred_pose, gt_pose) * self.w_mpjpe
        return dict(mpjpe_loss=mpjpe_loss)

    def get_heatmap_loss(self, pred_heatmap, gt_heatmap):
        loss_heatmap = self.criteria(pred_heatmap, gt_heatmap) * self.w_heatmap
        return loss_heatmap

    def evaluate_pose(self, pred_pose, gt_pose, prefix):
        B = pred_pose.shape[0]

        S1_hat = batch_compute_similarity_transform_numpy(pred_pose, gt_pose.to(dtype=torch.float))

        mpjpe = compute_mpjpe_batch(pred_pose, gt_pose) * self.cm2mm
        pa_mpjpe = compute_mpjpe_batch(S1_hat, gt_pose) * self.cm2mm
        pck_3d = compute_pck_3d_batch(pred_pose * self.cm2mm, gt_pose * self.cm2mm) * 100.0
        auc_3d = compute_auc_3d_batch(pred_pose * self.cm2mm, gt_pose * self.cm2mm) * 100.0

        metrics = OrderedDict()
        metrics[prefix+"_mpjpe"] = mpjpe.detach().cpu().numpy()
        metrics[prefix+"_pa_mpjpe"] = pa_mpjpe.detach().cpu().numpy()
        metrics[prefix+"_pck_3d"] = pck_3d.detach().cpu().numpy()
        metrics[prefix+"_auc_3d"] = auc_3d.detach().cpu().numpy()

        return metrics

    def evaluate_heatmap(self, pred_heatmap, gt_heatmap, prefix):
        B, V = pred_heatmap.size()[0:2]

        list_pred_heatmap = [pred_heatmap[:, i] for i in range(V)]
        list_pred_heatmap = [x.reshape(B, -1) for x in list_pred_heatmap]

        list_gt_heatmap = [gt_heatmap[:, i] for i in range(V)]
        list_gt_heatmap = [y.reshape(B, -1) for y in list_gt_heatmap]

        pos_inds = [y > 0 for y in list_gt_heatmap]

        error_heatmap = sum(torch.abs(x-y) for x, y in zip(list_pred_heatmap, list_gt_heatmap))
        error_heatmap = error_heatmap.sum(dim=1).reshape(B, ).detach().cpu()

        pos_error_heatmap = torch.tensor([
            sum(
                torch.abs(x[i][ind[i]] - y[i][ind[i]]).sum().detach().cpu()
                for x, y, ind in zip(list_pred_heatmap, list_gt_heatmap, pos_inds)
            )
            for i in range(B)
        ])

        metrics = OrderedDict()
        metrics[prefix+"_l1_error_heatmap"] = error_heatmap
        metrics[prefix+"_pos_l1_error_heatmap"] = pos_error_heatmap

        return metrics
