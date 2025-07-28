import numpy as np
import cv2
import json
from natsort import natsorted
import os, glob, argparse
from PIL import Image
from loguru import logger


def generate_target(joints, image_size=872, heatmap_size=64, num_joints=15, sigma=1):
    target_weight = np.ones((num_joints, 1), dtype=np.float32)

    target = np.zeros((num_joints,
                        heatmap_size,
                        heatmap_size),
                        dtype=np.float32)

    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        feat_stride = (image_size / heatmap_size, image_size / heatmap_size)
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)

        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size or ul[1] >= heatmap_size \
                or br[0] < 0 or br[1] < 0:
            target_weight[joint_id] = 0
            continue

        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        g_x = max(0, -ul[0]), min(br[0], heatmap_size) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size) - ul[1]
        img_x = max(0, ul[0]), min(br[0], heatmap_size)
        img_y = max(0, ul[1]), min(br[1], heatmap_size)

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target


def main(json_path, json_dir_name, list_joints):

    with open(json_path) as f:
        json_data = json.load(f)

    list_camera_name = ["camera_front_left", "camera_front_right", 'camera_back_left', 'camera_back_right']

    for camera_name in list_camera_name:

        data_jotin_pts2d = np.array([json_data["joints"][joint_name]["{}_pts2d".format(camera_name)] for joint_name in list_joints])

        heatmaps = generate_target(
            joints=data_jotin_pts2d,
            image_size=872,
            heatmap_size=64,
            num_joints=len(list_joints),
            sigma=1.0
        )

        npy_save_dir = os.path.join(
            os.path.dirname(json_path).replace(json_dir_name, "fisheye_hm"),
            camera_name,
            os.path.basename(json_path).replace(".json", ".npy")
        )

        if not os.path.exists(os.path.dirname(npy_save_dir)):
            os.makedirs(os.path.dirname(npy_save_dir))

        np.save(npy_save_dir, heatmaps)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="===")
    parser.add_argument(
        '--data_dir_path',
        type=str,
        default="/CT/datasets05/nobackup/Ego4View_rw",
        help="path to data directory"
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        choices=['rw', 'syn'],
        default='rw',
        help="type of dataset to use"
    )
    args = parser.parse_args()

    if args.dataset_type == 'rw':
        list_seq_dir_path = natsorted(glob.glob(os.path.join(args.data_dir_path, "2024*/S*/seq*")))
        json_dir_name = "json_smplx"
    elif args.dataset_type == 'syn':
        list_seq_dir_path = natsorted(glob.glob(os.path.join(args.data_dir_path, "rp*/*")))
        json_dir_name = "json_smplx_gendered"

    list_joints = [
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

    logger.debug("")
    logger.debug("++++++++++++++++")
    for seq_dir_path in list_seq_dir_path:
        logger.debug(seq_dir_path)
    logger.debug("++++++++++++++++")
    logger.debug("processing ...")
    logger.debug("")

    for seq_dir_path in list_seq_dir_path:
        list_json_path = natsorted(glob.glob(os.path.join(seq_dir_path, json_dir_name, "*.json")))

        for id, json_path in enumerate(list_json_path):
                main(json_path, json_dir_name, list_joints)

    logger.debug("")
    logger.debug("Finished !!!")