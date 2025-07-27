# Author: Hiroyasu Akada

import os
import glob
import random
from abc import ABCMeta

import numpy as np
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from loguru import logger


class Ego4ViewRWPose3DDataset(Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        data_root,
        info_json,
        action_id=0,
        pre_shuffle=False,
        **kwargs
    ):
        super(Ego4ViewRWPose3DDataset, self).__init__()

        self.data_root = data_root
        self.info_json = info_json
        self.action_id = int(action_id)

        self.dataset_kwargs = kwargs
        if self.dataset_kwargs["camera_pos"]:
            self.camera_pos = self.dataset_kwargs["camera_pos"]
        else:
            self.camera_pos == "all"

        self.frame_dataset = self.collect_dataset(info_json, pre_shuffle)

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

    def collect_dataset(self, info_json, pre_shuffle):
        data = []

        with open(info_json, 'r') as f:
            lines = f.readlines()

        for line_idx, line in enumerate(lines):

            line = line.strip()
            if len(line) == 0:
                continue

            list_frame_path = glob.glob(
                os.path.join(
                    self.data_root,
                    line,
                    "json_smplx",
                    "*.json"
                )
            )

            for frame_path in list_frame_path:
                data.append(frame_path)

        if pre_shuffle:
            random.shuffle(data)

        return data

    def load_data(self, idx):
        frame_path = self.frame_dataset[idx]

        # load pose data
        json_data = json.load(open(frame_path))
        gt_local_pose = np.array([
            json_data["joints"][joint_name]["device_pts3d"] for joint_name in self.list_joints
        ])
        pose_gt = gt_local_pose[:, :]  # [16, 3]

        # load device pose

        with open(os.path.join(frame_path.split("-")[0] + "_metadata.json"), 'r') as f:
            metadata = json.load(f)

        if self.camera_pos == "front":
            # load rgb data
            input_rgb_front_left_path = os.path.join(frame_path.replace("json_smplx", "fisheye_rgb/camera_front_left").replace(".json", ".png"))
            input_rgb_front_right_path = os.path.join(frame_path.replace("json_smplx", "fisheye_rgb/camera_front_right").replace(".json", ".png"))
            input_rgb_front_left = Image.open(input_rgb_front_left_path).convert("RGB")
            input_rgb_front_right = Image.open(input_rgb_front_right_path).convert("RGB")
            input_rgb_front_left = self.transform(input_rgb_front_left.resize([256, 256], Image.BICUBIC)).float().numpy()
            input_rgb_front_right = self.transform(input_rgb_front_right.resize([256, 256], Image.BICUBIC)).float().numpy()
            img_front_left = input_rgb_front_left[np.newaxis, :, :, :]  # [1, 3, 256, 256]
            img_front_right = input_rgb_front_right[np.newaxis, :, :, :]  # [1, 3, 256, 256]
            img = np.concatenate((img_front_left, img_front_right), axis=0)  # [2, 3, 256, 256]

            # load heatmap data
            gt_heatmap_front_left_path = os.path.join(frame_path.replace("json_smplx", "fisheye_hm/camera_front_left").replace(".json", ".npy"))
            gt_heatmap_front_right_path = os.path.join(frame_path.replace("json_smplx", "fisheye_hm/camera_front_right").replace(".json", ".npy"))
            gt_heatmap_front_left = np.load(gt_heatmap_front_left_path)
            gt_heatmap_front_right = np.load(gt_heatmap_front_right_path)
            heatmap_front_left = gt_heatmap_front_left[np.newaxis, 1:, :, :]  # [1, 15, 256, 256]
            heatmap_front_right = gt_heatmap_front_right[np.newaxis, 1:, :, :]  # [1, 15, 256, 256]
            heatmap = np.concatenate((heatmap_front_left, heatmap_front_right), axis=0)  # [2, 15, 256, 256]

            ct_mat_device_to_camera_front_left = np.asarray(metadata["coord_transformation_matrix"]["device_to_camera_front_left"])[np.newaxis, :, :]
            ct_mat_device_to_camera_front_right = np.asarray(metadata["coord_transformation_matrix"]["device_to_camera_front_right"])[np.newaxis, :, :]
            coord_trans_mat = np.concatenate((ct_mat_device_to_camera_front_left, ct_mat_device_to_camera_front_right), axis=0)

        elif self.camera_pos == "back":
            # load rgb data
            input_rgb_back_left_path = os.path.join(frame_path.replace("json_smplx", "fisheye_rgb/camera_back_left").replace(".json", ".png"))
            input_rgb_back_right_path = os.path.join(frame_path.replace("json_smplx", "fisheye_rgb/camera_back_right").replace(".json", ".png"))
            input_rgb_back_left = Image.open(input_rgb_back_left_path).convert("RGB")
            input_rgb_back_right = Image.open(input_rgb_back_right_path).convert("RGB")
            input_rgb_back_left = self.transform(input_rgb_back_left.resize([256, 256], Image.BICUBIC)).float().numpy()
            input_rgb_back_right = self.transform(input_rgb_back_right.resize([256, 256], Image.BICUBIC)).float().numpy()
            img_back_left = input_rgb_back_left[np.newaxis, :, :, :]  # [1, 3, 256, 256]
            img_back_right = input_rgb_back_right[np.newaxis, :, :, :]  # [1, 3, 256, 256]
            img = np.concatenate((img_back_left, img_back_right), axis=0)  # [2, 3, 256, 256]

            # load heatmap data
            gt_heatmap_back_left_path = os.path.join(frame_path.replace("json_smplx", "fisheye_hm/camera_back_left").replace(".json", ".npy"))
            gt_heatmap_back_right_path = os.path.join(frame_path.replace("json_smplx", "fisheye_hm/camera_back_right").replace(".json", ".npy"))
            gt_heatmap_back_left = np.load(gt_heatmap_back_left_path)
            gt_heatmap_back_right = np.load(gt_heatmap_back_right_path)
            heatmap_back_left = gt_heatmap_back_left[np.newaxis, 1:, :, :]  # [1, 15, 256, 256]
            heatmap_back_right = gt_heatmap_back_right[np.newaxis, 1:, :, :]  # [1, 15, 256, 256]
            heatmap = np.concatenate((heatmap_back_left, heatmap_back_right), axis=0)  # [2, 15, 256, 256]

            ct_mat_device_to_camera_back_left = np.asarray(metadata["coord_transformation_matrix"]["device_to_camera_back_left"])[np.newaxis, :, :]
            ct_mat_device_to_camera_back_right = np.asarray(metadata["coord_transformation_matrix"]["device_to_camera_back_right"])[np.newaxis, :, :]
            coord_trans_mat = np.concatenate((ct_mat_device_to_camera_back_left, ct_mat_device_to_camera_back_right), axis=0)

        else:
            # load rgb data
            input_rgb_front_left_path = os.path.join(frame_path.replace("json_smplx", "fisheye_rgb/camera_front_left").replace(".json", ".png"))
            input_rgb_front_right_path = os.path.join(frame_path.replace("json_smplx", "fisheye_rgb/camera_front_right").replace(".json", ".png"))
            input_rgb_back_left_path = os.path.join(frame_path.replace("json_smplx", "fisheye_rgb/camera_back_left").replace(".json", ".png"))
            input_rgb_back_right_path = os.path.join(frame_path.replace("json_smplx", "fisheye_rgb/camera_back_right").replace(".json", ".png"))
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
            gt_heatmap_front_left_path = os.path.join(frame_path.replace("json_smplx", "fisheye_hm/camera_front_left").replace(".json", ".npy"))
            gt_heatmap_front_right_path = os.path.join(frame_path.replace("json_smplx", "fisheye_hm/camera_front_right").replace(".json", ".npy"))
            gt_heatmap_back_left_path = os.path.join(frame_path.replace("json_smplx", "fisheye_hm/camera_back_left").replace(".json", ".npy"))
            gt_heatmap_back_right_path = os.path.join(frame_path.replace("json_smplx", "fisheye_hm/camera_back_right").replace(".json", ".npy"))
            gt_heatmap_front_left = np.load(gt_heatmap_front_left_path)
            gt_heatmap_front_right = np.load(gt_heatmap_front_right_path)
            gt_heatmap_back_left = np.load(gt_heatmap_back_left_path)
            gt_heatmap_back_right = np.load(gt_heatmap_back_right_path)
            heatmap_front_left = gt_heatmap_front_left[np.newaxis, 1:, :, :]  # [1, 15, 256, 256]
            heatmap_front_right = gt_heatmap_front_right[np.newaxis, 1:, :, :]  # [1, 15, 256, 256]
            heatmap_back_left = gt_heatmap_back_left[np.newaxis, 1:, :, :]  # [1, 15, 256, 256]
            heatmap_back_right = gt_heatmap_back_right[np.newaxis, 1:, :, :]  # [1, 15, 256, 256]
            heatmap = np.concatenate((heatmap_front_left, heatmap_front_right, heatmap_back_left, heatmap_back_right), axis=0)  # [4, 15, 256, 256]

            ct_mat_device_to_camera_front_left = np.asarray(metadata["coord_transformation_matrix"]["device_to_camera_front_left"])[np.newaxis, :, :]
            ct_mat_device_to_camera_front_right = np.asarray(metadata["coord_transformation_matrix"]["device_to_camera_front_right"])[np.newaxis, :, :]
            ct_mat_device_to_camera_back_left = np.asarray(metadata["coord_transformation_matrix"]["device_to_camera_back_left"])[np.newaxis, :, :]
            ct_mat_device_to_camera_back_right = np.asarray(metadata["coord_transformation_matrix"]["device_to_camera_back_right"])[np.newaxis, :, :]
            coord_trans_mat = np.concatenate((ct_mat_device_to_camera_front_left, ct_mat_device_to_camera_front_right, ct_mat_device_to_camera_back_left, ct_mat_device_to_camera_back_right), axis=0)

        ret_data = dict()
        ret_data["img"] = torch.from_numpy(img)
        ret_data["gt_heatmap"] = torch.from_numpy(heatmap)
        ret_data["gt_pose"] = torch.from_numpy(pose_gt)
        ret_data["coord_trans_mat"] = torch.from_numpy(coord_trans_mat)
        ret_data["frame_path"] = str(frame_path)

        return ret_data

    def __len__(self):
        return len(self.frame_dataset)

    def __getitem__(self, idx):
        return self.load_data(idx)
