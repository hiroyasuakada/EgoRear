# Author: Hiroyasu Akada

import os
import glob
from abc import ABCMeta

import numpy as np
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from loguru import logger


class Ego4ViewSynHeatmapMVFDataset(Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        data_root,
        info_json,
        action_id=0,
        **kwargs
    ):
        super(Ego4ViewSynHeatmapMVFDataset, self).__init__()

        self.data_root = data_root
        self.info_json = info_json
        self.action_id = int(action_id)

        self.dataset_kwargs = kwargs
        if self.dataset_kwargs["camera_pos"]:
            self.camera_pos = self.dataset_kwargs["camera_pos"]
        else:
            self.camera_pos == "all"

        self.frame_dataset = self.collect_dataset(info_json)

        transforms_list = []
        transforms_list.append(transforms.ToTensor())  # convert [0, 255] to [0, 1] for PIL images
        transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        self.transform = transforms.Compose(transforms_list)

        self.list_joints = [
            "Head",
            "Neck",
            "LeftArm",
            "RightArm",
            "LeftForeArm",
            "RightForeArm",
            "LeftHand",
            "RightHand",
            "LeftUpLeg",
            "RightUpLeg",
            "LeftLeg",
            "RightLeg",
            "LeftFoot",
            "RightFoot",
            "LeftToeBase",
            "RightToeBase",
        ]

    def collect_dataset(self, info_json):
        data = []

        with open(info_json, 'r') as f:
            lines = f.readlines()

        for line_idx, line in enumerate(lines):

            line = line.strip()
            if len(line) == 0:
                continue

            list_seq_path = glob.glob(os.path.join(self.data_root, line, "*"))

            for seq_path in list_seq_path:

                list_frame_path = glob.glob(
                    os.path.join(
                        seq_path,
                        "json_smplx_gendered",
                        "*.json"
                    )
                )

                for frame_path in list_frame_path:
                    data.append(frame_path)

        return data

    def load_data(self, idx):
        frame_path = self.frame_dataset[idx]

        if self.camera_pos == "front":
            # load rgb data
            input_rgb_front_left_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_rgb/camera_front_left").replace(".json", ".jpg"))
            input_rgb_front_right_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_rgb/camera_front_right").replace(".json", ".jpg"))
            input_rgb_front_left = Image.open(input_rgb_front_left_path).convert("RGB")
            input_rgb_front_right = Image.open(input_rgb_front_right_path).convert("RGB")
            input_rgb_front_left = self.transform(input_rgb_front_left.resize([256, 256], Image.BICUBIC)).float().numpy()
            input_rgb_front_right = self.transform(input_rgb_front_right.resize([256, 256], Image.BICUBIC)).float().numpy()
            img_front_left = input_rgb_front_left[np.newaxis, :, :, :]  # [1, 3, 256, 256]
            img_front_right = input_rgb_front_right[np.newaxis, :, :, :]  # [1, 3, 256, 256]
            img = np.concatenate((img_front_left, img_front_right), axis=0)  # [2, 3, 256, 256]

            # load heatmap data
            gt_heatmap_front_left_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_hm/camera_front_left").replace(".json", ".npy"))
            gt_heatmap_front_right_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_hm/camera_front_right").replace(".json", ".npy"))
            gt_heatmap_front_left = np.load(gt_heatmap_front_left_path)
            gt_heatmap_front_right = np.load(gt_heatmap_front_right_path)
            heatmap_front_left = gt_heatmap_front_left[np.newaxis, 1:, :, :]  # [1, 15, 256, 256]
            heatmap_front_right = gt_heatmap_front_right[np.newaxis, 1:, :, :]  # [1, 15, 256, 256]
            heatmap = np.concatenate((heatmap_front_left, heatmap_front_right), axis=0)  # [2, 15, 256, 256]

        elif self.camera_pos == "back":
            # load rgb data
            input_rgb_back_left_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_rgb/camera_back_left").replace(".json", ".jpg"))
            input_rgb_back_right_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_rgb/camera_back_right").replace(".json", ".jpg"))
            input_rgb_back_left = Image.open(input_rgb_back_left_path).convert("RGB")
            input_rgb_back_right = Image.open(input_rgb_back_right_path).convert("RGB")
            input_rgb_back_left = self.transform(input_rgb_back_left.resize([256, 256], Image.BICUBIC)).float().numpy()
            input_rgb_back_right = self.transform(input_rgb_back_right.resize([256, 256], Image.BICUBIC)).float().numpy()
            img_back_left = input_rgb_back_left[np.newaxis, :, :, :]  # [1, 3, 256, 256]
            img_back_right = input_rgb_back_right[np.newaxis, :, :, :]  # [1, 3, 256, 256]
            img = np.concatenate((img_back_left, img_back_right), axis=0)  # [2, 3, 256, 256]

            # load heatmap data
            gt_heatmap_back_left_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_hm/camera_back_left").replace(".json", ".npy"))
            gt_heatmap_back_right_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_hm/camera_back_right").replace(".json", ".npy"))
            gt_heatmap_back_left = np.load(gt_heatmap_back_left_path)
            gt_heatmap_back_right = np.load(gt_heatmap_back_right_path)
            heatmap_back_left = gt_heatmap_back_left[np.newaxis, 1:, :, :]  # [1, 15, 256, 256]
            heatmap_back_right = gt_heatmap_back_right[np.newaxis, 1:, :, :]  # [1, 15, 256, 256]
            heatmap = np.concatenate((heatmap_back_left, heatmap_back_right), axis=0)  # [2, 15, 256, 256]

        else:
            # load rgb data
            input_rgb_front_left_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_rgb/camera_front_left").replace(".json", ".jpg"))
            input_rgb_front_right_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_rgb/camera_front_right").replace(".json", ".jpg"))
            input_rgb_back_left_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_rgb/camera_back_left").replace(".json", ".jpg"))
            input_rgb_back_right_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_rgb/camera_back_right").replace(".json", ".jpg"))
            input_rgb_front_left = Image.open(input_rgb_front_left_path).convert("RGB")
            input_rgb_front_right = Image.open(input_rgb_front_right_path).convert("RGB")
            input_rgb_back_left = Image.open(input_rgb_back_left_path).convert("RGB")
            input_rgb_back_right = Image.open(input_rgb_back_right_path).convert("RGB")
            input_rgb_front_left = self.transform(input_rgb_front_left.resize([256, 256], Image.BICUBIC)).float().numpy()
            input_rgb_front_right = self.transform(input_rgb_front_right.resize([256, 256], Image.BICUBIC)).float().numpy()
            input_rgb_back_left = self.transform(input_rgb_back_left.resize([256, 256], Image.BICUBIC)).float().numpy()
            input_rgb_back_right = self.transform(input_rgb_back_right.resize([256, 256], Image.BICUBIC)).float().numpy()
            img_front_left = input_rgb_front_left[np.newaxis, :, :, :]  # [1, 3, 256, 256]
            img_front_right = input_rgb_front_right[np.newaxis, :, :, :]  # [1, 3, 256, 256]
            img_back_left = input_rgb_back_left[np.newaxis, :, :, :]  # [1, 3, 256, 256]
            img_back_right = input_rgb_back_right[np.newaxis, :, :, :]  # [1, 3, 256, 256]
            img = np.concatenate((img_front_left, img_front_right, img_back_left, img_back_right), axis=0)  # [4, 3, 256, 256]

            # load heatmap data
            gt_heatmap_front_left_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_hm/camera_front_left").replace(".json", ".npy"))
            gt_heatmap_front_right_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_hm/camera_front_right").replace(".json", ".npy"))
            gt_heatmap_back_left_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_hm/camera_back_left").replace(".json", ".npy"))
            gt_heatmap_back_right_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_hm/camera_back_right").replace(".json", ".npy"))
            gt_heatmap_front_left = np.load(gt_heatmap_front_left_path)
            gt_heatmap_front_right = np.load(gt_heatmap_front_right_path)
            gt_heatmap_back_left = np.load(gt_heatmap_back_left_path)
            gt_heatmap_back_right = np.load(gt_heatmap_back_right_path)
            heatmap_front_left = gt_heatmap_front_left[np.newaxis, 1:, :, :]  # [1, 15, 256, 256]
            heatmap_front_right = gt_heatmap_front_right[np.newaxis, 1:, :, :]  # [1, 15, 256, 256]
            heatmap_back_left = gt_heatmap_back_left[np.newaxis, 1:, :, :]  # [1, 15, 256, 256]
            heatmap_back_right = gt_heatmap_back_right[np.newaxis, 1:, :, :]  # [1, 15, 256, 256]
            heatmap = np.concatenate((heatmap_front_left, heatmap_front_right, heatmap_back_left, heatmap_back_right), axis=0)  # [4, 15, 256, 256]

        ret_data = dict()
        ret_data["img"] = torch.from_numpy(img)
        ret_data["gt_heatmap"] = torch.from_numpy(heatmap)
        ret_data["frame_path"] = str(frame_path)

        return ret_data

    def __len__(self):
        return len(self.frame_dataset)

    def __getitem__(self, idx):
        return self.load_data(idx)
