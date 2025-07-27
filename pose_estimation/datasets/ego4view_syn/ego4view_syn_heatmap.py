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



class Ego4ViewSynHeatmapDataset(Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        data_root,
        info_json,
        **kwargs
    ):
        super(Ego4ViewSynHeatmapDataset, self).__init__()

        self.data_root = data_root
        self.info_json = info_json

        self.dataset_kwargs = kwargs
        self.camera_pos = self.dataset_kwargs["camera_pos"]

        self.dataset = self.collect_dataset()

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

    def collect_dataset(self):
        data = []

        with open(self.info_json) as f:
            lines = f.readlines()

        for line_idx, line in enumerate(lines[0:1]):

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

                    if (self.camera_pos == "front") or (self.camera_pos == "all"):
                        data.append((frame_path, 0))  # front left
                        data.append((frame_path, 1))  # front right

                    elif (self.camera_pos == "back") or (self.camera_pos == "all"):
                        data.append((frame_path, 2))  # back left
                        data.append((frame_path, 3))  # back right

                    else:
                        1/0

        return data

    def load_data(self, idx):
        frame_path, view = self.dataset[idx]

        if view == 0:
            input_rgb_front_left_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_rgb/camera_front_left").replace(".json", ".jpg"))
            input_rgb_front_left = Image.open(input_rgb_front_left_path).convert("RGB")
            input_rgb_front_left = self.transform(input_rgb_front_left.resize([256, 256], Image.BICUBIC)).float().numpy()
            img = input_rgb_front_left[np.newaxis, :, :, :]  # [1, 3, 256, 256]

            gt_heatmap_front_left_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_hm/camera_front_left").replace(".json", ".npy"))
            gt_heatmap_front_left = np.load(gt_heatmap_front_left_path)
            heatmap = gt_heatmap_front_left[np.newaxis, 1:, :, :]  # [16, 64, 64] to [15, 64, 64] without head

        elif view == 1:
            input_rgb_front_right_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_rgb/camera_front_right").replace(".json", ".jpg"))
            input_rgb_front_right = Image.open(input_rgb_front_right_path).convert("RGB")
            input_rgb_front_right = self.transform(input_rgb_front_right.resize([256, 256], Image.BICUBIC)).float().numpy()
            img = input_rgb_front_right[np.newaxis, :, :, :]  # [1, 3, 256, 256]

            gt_heatmap_front_right_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_hm/camera_front_right").replace(".json", ".npy"))
            gt_heatmap_front_right = np.load(gt_heatmap_front_right_path)
            heatmap = gt_heatmap_front_right[np.newaxis, 1:, :, :]  # [16, 64, 64] to [15, 64, 64] without head

        elif view == 2:
            input_rgb_back_left_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_rgb/camera_back_left").replace(".json", ".jpg"))
            input_rgb_back_left = Image.open(input_rgb_back_left_path).convert("RGB")
            input_rgb_back_left = self.transform(input_rgb_back_left.resize([256, 256], Image.BICUBIC)).float().numpy()
            img = input_rgb_back_left[np.newaxis, :, :, :]  # [1, 3, 256, 256]

            gt_heatmap_back_left_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_hm/camera_back_left").replace(".json", ".npy"))
            gt_heatmap_back_left = np.load(gt_heatmap_back_left_path)
            heatmap = gt_heatmap_back_left[np.newaxis, 1:, :, :]  # [16, 64, 64] to [15, 64, 64] without head

        elif view == 3:
            input_rgb_back_right_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_rgb/camera_back_right").replace(".json", ".jpg"))
            input_rgb_back_right = Image.open(input_rgb_back_right_path).convert("RGB")
            input_rgb_back_right = self.transform(input_rgb_back_right.resize([256, 256], Image.BICUBIC)).float().numpy()
            img = input_rgb_back_right[np.newaxis, :, :, :]  # [1, 3, 256, 256]

            gt_heatmap_back_right_path = os.path.join(frame_path.replace("json_smplx_gendered", "fisheye_hm/camera_back_right").replace(".json", ".npy"))
            gt_heatmap_back_right = np.load(gt_heatmap_back_right_path)
            heatmap = gt_heatmap_back_right[np.newaxis, 1:, :, :]  # [16, 64, 64] to [15, 64, 64] without head

        ret_data = dict()
        ret_data["img"] = torch.from_numpy(img)
        ret_data["gt_heatmap"] = torch.from_numpy(heatmap)
        ret_data["frame_path"] = str(frame_path)
        ret_data["view"] = str(view)

        return ret_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.load_data(idx)

