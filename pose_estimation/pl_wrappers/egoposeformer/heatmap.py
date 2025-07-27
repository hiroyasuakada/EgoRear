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

from pose_estimation.models.estimator import EgoPoseFormerHeatmap
from pose_estimation.datasets.dataset import get_dataset

from pose_estimation.utils.state_dict import fix_model_state_dict
from pose_estimation.utils.loss import get_max_preds

from loguru import logger



class PoseHeatmapLightningModel(LightningModule):
    def __init__(
        self,
        model_cfg: dict,
        dataset_type: str,
        data_root: str,
        lr: float,
        weight_decay: float,
        lr_decay_epochs: tuple,
        warmup_iters: int,
        w_heatmap: float,
        batch_size: int,
        workers: int,
        compile: bool,
        compile_mode: str,
        save_result: bool,
        network_pretrained: str = None,
        dataset_kwargs: dict = {}
    ):
        super().__init__()

        self.save_hyperparameters()

        self.dataset_type = dataset_type
        self.dataset_kwargs = dataset_kwargs

        self.compile = compile
        self.compile_mode = compile_mode

        self.network = EgoPoseFormerHeatmap(**model_cfg)

        if network_pretrained != None:
            self.load_trained_model(self.network, network_pretrained)

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_epochs = lr_decay_epochs
        self.warmup_iters = warmup_iters
        self.w_heatmap = w_heatmap
        self.num_heatmap = self.network.num_heatmap

        self.data_root = data_root
        self.batch_size = batch_size
        self.workers = workers

        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None

        self.criteria = nn.MSELoss(reduction="mean")

        self.log_img_every_n_batches = 2500

        self.save_result = save_result
        self.save_result_every_n_batches = 2

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
        B, V, img_c, img_h, img_w = img.shape

        pred_heatmap = self.network(img)

        list_pred_heatmap = [pred_heatmap[:, i] for i in range(V)]
        list_gt_heatmap = [gt_heatmap[:, i] for i in range(V)]
        loss = sum([self.get_loss(x, y) for x, y in zip(list_pred_heatmap, list_gt_heatmap)])
        loss_dict = dict(heatmap_loss=loss)
        loss_total = loss_dict.get("heatmap_loss")
        self.log("train/heatmap_loss", loss_total)

        return loss_total

    def validation_step(self, batch, batch_idx):
        self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.eval_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        if (batch_idx != 0) and (batch_idx % self.save_result_every_n_batches == 0):
            self.eval_step(batch, batch_idx, "predict")

        if (batch_idx != 0) and (batch_idx % (self.save_result_every_n_batches) == 1):
            self.eval_step(batch, batch_idx, "predict")

    def eval_step(self, batch, batch_idx, mode):
        self.network.eval()

        img = batch["img"]
        gt_heatmap = batch["gt_heatmap"]
        B, V, img_c, img_h, img_w = img.shape

        pred_heatmap = self.network(img)

        if (mode == "val") or (mode == "test"):
            output_dict = OrderedDict()
            metric_pred_init = self.evaluate(pred_heatmap, gt_heatmap, "proposal")
            output_dict.update(metric_pred_init)

            for k, v in output_dict.items():
                self.log("{}/{}".format(mode, k), v.mean(), sync_dist=True)

        return None

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure,):
        optimizer.step(closure=optimizer_closure)
        if self.trainer.global_step < self.warmup_iters:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.warmup_iters))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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

    def get_loss(self, pred_heatmap, gt_heatmap):
        loss_heatmap = self.criteria(pred_heatmap, gt_heatmap)
        loss_heatmap *= self.w_heatmap
        return loss_heatmap

    def evaluate(self, pred_heatmap, gt_heatmap, prefix):
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

        mse_heatmap = self.criteria(pred_heatmap, gt_heatmap)

        pred_heatmap_pts2d, _, _ = self.get_anchors_2d_from_hm(pred_heatmap)
        gt_heatmap_pts2d, gt_heatmap_maxvals, gt_heatmap_mask_valid = self.get_anchors_2d_from_hm(gt_heatmap)
        mse_pts2d = self.criteria(pred_heatmap_pts2d * gt_heatmap_mask_valid.unsqueeze(-1), gt_heatmap_pts2d * gt_heatmap_mask_valid.unsqueeze(-1))

        metrics = OrderedDict()
        metrics[prefix+"_l1_error_heatmap"] = error_heatmap
        metrics[prefix+"_pos_l1_error_heatmap"] = pos_error_heatmap
        metrics[prefix+"_mse_heatmap"] = mse_heatmap
        metrics[prefix+"_mse_pts2d"] = mse_pts2d

        return metrics


    def get_anchors_2d_from_hm(self, heatmap):

        with torch.no_grad():
            B, V, C, H, W = heatmap.shape
            heatmap = heatmap.view(B * V, C, H, W)

            pts2d, maxvals, mask_valid = get_max_preds(
                heatmap, threshold=1.0, normalize=False
            )

            pts2d = pts2d.view(B, V, self.num_heatmap, 2)
            maxvals = maxvals.view(B, V, self.num_heatmap)
            mask_valid = mask_valid.view(B, V, self.num_heatmap)

        return pts2d, maxvals, mask_valid