import numpy as np
import open3d
from pose_estimation.utils.pose_visualization_utils import get_cylinder, get_sphere
from scipy.io import loadmat
import cv2
import os
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter1d
from loguru import logger

class Skeleton:
    heatmap_sequence = [
            "head",
            "neck_01",
            "upperarm_l",
            "upperarm_r",
            "lowerarm_l",
            "lowerarm_r",
            "hand_l",
            "hand_r",
            "thigh_l",
            "thigh_r",
            "calf_l",
            "calf_r",
            "foot_l",
            "foot_r",
            "ball_l",
            "ball_r"
        ]
    lines = [
        (0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),
        (2, 8), (3, 9), (8, 10), (9, 11), (10, 12), (11, 13), (12, 14), (13, 15),
        (8, 9)
    ]
    kinematic_ids =     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    kinematic_parents = [0, 0, 1, 1, 2, 3, 4, 5, 2, 3, 8, 9, 10, 11, 12, 13]


    list_line_color = [
        (0, 1, (204, 0, 0)),
        (1, 2, (255, 51, 0)),
        (1, 3, (255, 51, 0)),
        (2, 4, (255, 153, 0)),
        (3, 5, (0, 102, 0)),
        (4, 6, (255,255,51)),
        (5, 7, (0, 255, 0)),
        (2, 8, (153, 102, 153)),
        (3, 9, (153, 102, 153)),
        (8, 10, (0, 153, 255)),
        (9, 11, (255, 51, 255)),
        (10, 12, (0, 102, 255)),
        (11, 13, (51, 51, 255)),
        (12, 14, (0, 51, 153)),
        (13, 15, (0, 0, 255)),
        (8, 9, (120, 100, 255))
    ]

    list_joint_color = [
            ("head", (204, 0, 0)),
            ("neck_01", (255, 51, 51)),
            ("upperarm_l", (255, 153, 0)),
            ("upperarm_r", (0, 102, 0)),
            ("lowerarm_l", (255, 255, 51)),
            ("lowerarm_r", (0, 255, 0)),
            ("hand_l", (255,255,51)),
            ("hand_r", (0, 255, 0)),
            ("thigh_l", (0, 153, 255)),
            ("thigh_r", (255, 51, 255)),
            ("calf_l", (0, 102, 255)),
            ("calf_r", (51, 51, 255)),
            ("foot_l", (0, 51, 153)),
            ("foot_r", (0, 0, 255)),
            ("ball_l", (0, 51, 153)),
            ("ball_r", (0, 0, 255))
        ]


    def __init__(self, calibration_path):
        self.skeleton = None
        self.skeleton_mesh = None

    def set_skeleton(self, heatmap, depth, bone_length=None, to_mesh=True):
        heatmap = np.expand_dims(heatmap, axis=0)
        preds, _ = self.get_max_preds(heatmap)
        pred = preds[0]

        points_3d = self.camera.camera2world(pred, depth)

        if bone_length is not None:
            points_3d = self._skeleton_resize(points_3d, bone_length)
        self.skeleton = points_3d
        if to_mesh:
            self.skeleton_to_mesh()
        return self.skeleton

    def joints_2_mesh(self, joints_3d, color=None, colorful_pose=False):
        self.skeleton = joints_3d
        self.skeleton_to_mesh(color=color, colorful_pose=colorful_pose)
        skeleton_mesh = self.skeleton_mesh
        self.skeleton_mesh = None
        self.skeleton = None
        return skeleton_mesh

    def joint_list_2_mesh_list(self, joints_3d_list):
        mesh_list = []
        for joints_3d in joints_3d_list:
            mesh_list.append(self.joints_2_mesh(joints_3d))
        return mesh_list

    def get_skeleton_mesh(self):
        if self.skeleton_mesh is None:
            raise Exception("Skeleton is not prepared.")
        else:
            return self.skeleton_mesh

    def save_skeleton_mesh(self, out_path):
        if self.skeleton_mesh is None:
            raise Exception("Skeleton is not prepared.")
        else:
            open3d.io.write_triangle_mesh(out_path, mesh=self.skeleton_mesh)

    def set_skeleton_from_file(self, heatmap_file, depth_file, bone_length_file=None, to_mesh=True):
        if bone_length_file is not None:
            bone_length_mat = loadmat(bone_length_file)
            mean3D = bone_length_mat['mean3D'].T
            bones_mean = mean3D - mean3D[self.kinematic_parents, :]
            bone_length = np.linalg.norm(bones_mean, axis=1)
        else:
            bone_length = None
        heatmap_mat = loadmat(heatmap_file)
        depth_mat = loadmat(depth_file)
        depth = depth_mat['depth'][0]
        heatmap = heatmap_mat['heatmap']
        heatmap = cv2.resize(heatmap, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)
        heatmap = np.pad(heatmap, ((0, 0), (128, 128), (0, 0)), 'constant', constant_values=0)
        heatmap = heatmap.transpose((2, 0, 1))
        return self.set_skeleton(heatmap, depth, bone_length, to_mesh)

    def skeleton_resize_seq(self, joint_list, bone_length_file):
        bone_length_mat = loadmat(bone_length_file)
        mean3D = bone_length_mat['mean3D'].T
        bones_mean = mean3D - mean3D[self.kinematic_parents, :]
        bone_length = np.linalg.norm(bones_mean, axis=1)

        for i in range(len(joint_list)):
            joint_list[i] = self._skeleton_resize(joint_list[i], bone_length)
        return joint_list

    def skeleton_resize_single(self, joint, bone_length_file):
        bone_length_mat = loadmat(bone_length_file)
        mean3D = bone_length_mat['mean3D'].T
        bones_mean = mean3D - mean3D[self.kinematic_parents, :]
        bone_length = np.linalg.norm(bones_mean, axis=1)

        joint = self._skeleton_resize(joint, bone_length)
        return joint

    def skeleton_resize_standard_skeleton(self, joint_input, joint_standard):
        bones_mean = joint_standard - joint_standard[self.kinematic_parents, :]
        bone_length = np.linalg.norm(bones_mean, axis=1) * 1000.

        joint = self._skeleton_resize(joint_input, bone_length)
        return joint

    def _skeleton_resize(self, points_3d, bone_length):
        estimated_bone_vec = points_3d - points_3d[self.kinematic_parents, :]
        estimated_bone_length = np.linalg.norm(estimated_bone_vec, axis=1)
        multi = bone_length[1:] / estimated_bone_length[1:]
        multi = np.concatenate(([0], multi))
        multi = np.stack([multi] * 3, axis=1)
        resized_bones_vec = estimated_bone_vec * multi / 1000

        joints_rescaled = points_3d
        for i in range(joints_rescaled.shape[0]):
            joints_rescaled[i, :] = joints_rescaled[self.kinematic_parents[i], :] + resized_bones_vec[i, :]
        return joints_rescaled

    def render(self):
        mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        open3d.visualization.draw_geometries([self.skeleton_mesh, mesh_frame])

    def skeleton_to_mesh(self, color=None, colorful_pose=False):

        final_mesh = open3d.geometry.TriangleMesh()

        for i in range(len(self.skeleton)):

            if colorful_pose:
                joint_color = self.list_joint_color[i][1]
                joint_color = [joint_color[2] / 255.0, joint_color[1] / 255.0, joint_color[0] / 255.0]
                keypoint_mesh = get_sphere(position=self.skeleton[i], radius=2, color=joint_color)
            elif color:
                keypoint_mesh = get_sphere(position=self.skeleton[i], radius=2, color=color)
            else:
                keypoint_mesh = get_sphere(position=self.skeleton[i], radius=0.02)

            final_mesh = final_mesh + keypoint_mesh

        for i, line in enumerate(self.lines):
            line_start_i = line[0]
            line_end_i = line[1]

            start_point = self.skeleton[line_start_i]
            end_point = self.skeleton[line_end_i]

            if colorful_pose:
                line_color = self.list_line_color[i][2]
                line_color = [line_color[2] / 255.0, line_color[1] / 255.0, line_color[0] / 255.0]
                line_mesh = get_cylinder(start_point, end_point, radius=0.5, color=line_color)
            elif color:
                line_mesh = get_cylinder(start_point, end_point, radius=0.5, color=color)
            else:
                line_mesh = get_cylinder(start_point, end_point, radius=0.005)

            final_mesh += line_mesh
        self.skeleton_mesh = final_mesh


        return final_mesh

    def smooth(self, pose_sequence, sigma):
        pose_sequence = np.asarray(pose_sequence)
        pose_sequence_result = np.zeros_like(pose_sequence)
        keypoint_num = pose_sequence.shape[1]
        for i in range(keypoint_num):
            pose_sequence_i = pose_sequence[:, i, :]
            pose_sequence_filtered = gaussian_filter1d(pose_sequence_i, sigma, axis=0)
            pose_sequence_result[:, i, :] = pose_sequence_filtered
        return pose_sequence_result

    def get_max_preds(self, batch_heatmaps):
        assert isinstance(batch_heatmaps, np.ndarray), \
            'batch_heatmaps should be numpy.ndarray'
        assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds, maxvals