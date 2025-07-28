# Author: Hiroyasu Akada

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import open3d
import json
import cv2
from scipy.spatial.transform import Rotation
from copy import deepcopy

# from utils.skeleton import Skeleton
# skeleton_model = Skeleton(calibration_path='utils/fisheye/fisheye_calib_unrealego_scaramuzza.json')

# convert a tensor into a numpy array
def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8, is_depth=False, is_heatmap=False, is_video=False):
    if image_tensor.dim() == 3:  # (C, H, W)
        image_tensor = image_tensor.cpu().float()
    else: # (B, C, H, W) -> (C, H, W)
        image_tensor = image_tensor[0].cpu().float()

    # size (S, C, H, W) -> (C, H, W) of the last frame of the sequence
    if is_video:
        image_tensor = image_tensor[-1]

    if is_depth:
        image_tensor = image_tensor * bytes
    elif is_heatmap:
        image_tensor = torch.clamp(torch.sum(image_tensor, dim=0, keepdim=True), min=0.0, max=1.0) * bytes
    else:
        # image_tensor = (image_tensor + 1.0) / 2.0 * bytes
        image_tensor = denormalize_ImageNet(image_tensor) * bytes

    image_numpy = (image_tensor.permute(1, 2, 0)).detach().numpy().astype(imtype)
    return image_numpy

def tensor2pose(joints_3d, is_video=False):
    # size (B, S, num of heatmaps, 3) -> (S, num of heatmaps, 3)
    joints_3d = joints_3d[0].cpu().float()

    # size (S, num of heatmaps, 3) -> (num of heatmaps, 3) of the last frame of the sequence
    if is_video:
        joints_3d = joints_3d[-1]

    return joints_3d

def tensor2meshpose(joints_3d, color, save_dir, name, step, is_video=False):
    # size (B, S, num of heatmaps, 3) -> (S, num of heatmaps, 3)
    if torch.is_tensor(joints_3d):
        joints_3d = joints_3d[0].cpu().float()

    # size (S, num of heatmaps, 3) -> (num of heatmaps, 3) of the last frame of the sequence
    if is_video:
        joints_3d = joints_3d[-1]


    mesh = skeleton_model.joints_2_mesh(joints_3d, color)

    path_dir = os.path.join(save_dir, name)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    path_to_mesh = os.path.join(path_dir, "{}.ply".format(step))
    open3d.io.write_triangle_mesh(path_to_mesh, mesh)


    mesh_colorful = skeleton_model.joints_2_mesh(joints_3d, colorful_pose=True)

    path_dir_colorful = os.path.join(save_dir, name + "_c")
    if not os.path.exists(path_dir_colorful):
        os.makedirs(path_dir_colorful)

    path_to_mesh_colorful = os.path.join(path_dir_colorful, "{}.ply".format(step))
    open3d.io.write_triangle_mesh(path_to_mesh_colorful, mesh_colorful)

    return mesh


def apply_softmax(heatmap):
    """
    Compute the soft-argmax of a heatmap.
    Args:
        heatmap: Input heatmap of shape (H, W).
    Returns:
        Coordinates (y, x) of the soft-argmax.
    """
    H, W = heatmap.shape
    y_indices, x_indices = np.indices((H, W))
    normalized_heatmap = np.exp(heatmap) / np.sum(np.exp(heatmap))
    y = np.sum(y_indices * normalized_heatmap)
    x = np.sum(x_indices * normalized_heatmap)
    return x, y


def integrate_tensor_2d(heatmaps, softmax=True, multiplier=100.0):
    """Applies softmax to heatmaps and integrates them to get their's "center of masses"

    Args:
        heatmaps torch tensor of shape (batch_size, n_heatmaps, h, w): input heatmaps

    Returns:
        coordinates torch tensor of shape (batch_size, n_heatmaps, 2): coordinates of center of masses of all heatmaps

    URL: https://github.com/karfly/learnable-triangulation-pytorch/blob/master/mvn/utils/op.py

    """
    batch_size, n_heatmaps, h, w = heatmaps.shape

    heatmaps = heatmaps * multiplier

    heatmaps = heatmaps.reshape((batch_size, n_heatmaps, -1))
    if softmax:
        heatmaps = nn.functional.softmax(heatmaps, dim=2)
    else:
        heatmaps = nn.functional.relu(heatmaps)

    heatmaps = heatmaps.reshape((batch_size, n_heatmaps, h, w))

    mass_x = heatmaps.sum(dim=2)
    mass_y = heatmaps.sum(dim=3)

    mass_times_coord_x = mass_x * torch.arange(w).type(torch.float).to(mass_x.device)
    mass_times_coord_y = mass_y * torch.arange(h).type(torch.float).to(mass_y.device)

    x = mass_times_coord_x.sum(dim=2, keepdim=True)
    y = mass_times_coord_y.sum(dim=2, keepdim=True)

    if not softmax:
        x = x / mass_x.sum(dim=2, keepdim=True)
        y = y / mass_y.sum(dim=2, keepdim=True)

    coordinates = torch.cat((x, y), dim=2)
    coordinates = coordinates.reshape((batch_size, n_heatmaps, 2))

    return coordinates, heatmaps


# def tensor2pose(joints_3d, color, save_dir, name, step):
#     joints_3d = joints_3d[0].cpu().float()

#     jsn = json.load(open("../../../../../CT/UnrealEgo/static00/UnrealEgoData_visualization/KyotoAlley_Showcase_Visualization/Day/rp_manuel_rigged_001_ue4/SKM_MenReadingGlasses_Shape_01/023/3DPeople_rp_manuel_animated_001_dancing_ue4/json/{}.json".format(step)))

#     joints_3d = joints_3d + torch.tensor(jsn["joints"]["pelvis"]["camera_left_pts3d"]) - torch.tensor([-15, 10, 0]) # -jsn["camera_left"]["trans"] origin-relative

#     mesh = skeleton_model.joints_2_mesh(joints_3d, color)

#     path_dir = os.path.join(save_dir, name)
#     if not os.path.exists(path_dir):
#         os.makedirs(path_dir)

#     path_to_mesh = os.path.join(path_dir, "{}.ply".format(step))
#     # print(path_to_mesh)
#     open3d.io.write_triangle_mesh(path_to_mesh, mesh)

#     return mesh


def denormalize_ImageNet(x):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return x * std + mean

def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])
    image_numpy = Image.fromarray(image_numpy)
    image_numpy.save(image_path)
    # imageio.imwrite(image_path, image_numpy)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def draw_2d_joints(joints, img):

    # OpencvImage_array = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img = OpencvImage_array

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

    img_new = deepcopy(img)

    joints_num = joints.shape[0]
    for line in list_line_color:
        if line[0] < joints_num and line[1] < joints_num:
            start = joints[line[0]].astype(np.int32)
            end = joints[line[1]].astype(np.int32)
            paint_color = line[2]
            img_new = cv2.line(img_new, (start[0], start[1]), (end[0], end[1]), color=paint_color, thickness=5)
    for j in range(joints_num):
        img_new = cv2.circle(img_new, center=(joints[j][0].astype(np.int32), joints[j][1].astype(np.int32)),
                        radius=2, color=list_joint_color[j][1], thickness=10)

    return img_new

# def draw_2d_joints(joints, img, left_color=(0, 255, 0), right_color=(255, 0, 0), center_color=(0, 0, 255)):

#     # OpencvImage_array = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     # img = OpencvImage_array

#     list_line_color = [
#         (0, 1, "center"),
#         (1, 2, "left"),
#         (1, 3, "right"),
#         (2, 4, "left"),
#         (3, 5, "right"),
#         (4, 6, "left"),
#         (5, 7, "right"),
#         (2, 8, "left"),
#         (3, 9, "right"),
#         (8, 10, "left"),
#         (9, 11, "right"),
#         (10, 12, "left"),
#         (11, 13, "right"),
#         (12, 14, "left"),
#         (13, 15, "right"),
#         # (8, 9)
#     ]

#     joints_num = joints.shape[0]
#     for line in list_line_color:
#         if line[0] < joints_num and line[1] < joints_num:
#             start = joints[line[0]].astype(np.int32)
#             end = joints[line[1]].astype(np.int32)
#             left_or_right = line[2]
#             if left_or_right == 'right':
#                 paint_color = right_color
#             elif left_or_right == "left":
#                 paint_color = left_color
#             else:
#                 paint_color = center_color
#             img = cv2.line(img, (start[0], start[1]), (end[0], end[1]), color=paint_color, thickness=4)
#     for j in range(joints_num):
#         img = cv2.circle(img, center=(joints[j][0].astype(np.int32), joints[j][1].astype(np.int32)),
#                         radius=2, color=(0, 0, 255), thickness=6)

#     return img


# # convert a tensor into a numpy array
# def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8, depth=False, denorm_ImageNet=False):
#     if image_tensor.dim() == 3:  # (C, H, W)
#         image_tensor = image_tensor.cpu().float()
#     else:
#         image_tensor = image_tensor[0].cpu().float()

#     if depth:
#         image_tensor = image_tensor * bytes
#     # elif denorm_ImageNet:
#     #     image_tensor = denormalize_ImageNet(image_tensor) * bytes
#     else:
#         image_tensor = (image_tensor + 1.0) / 2.0 * bytes

#     image_numpy = (image_tensor.permute(1, 2, 0)).numpy().astype(imtype)
#     return image_numpy

# def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8):
#     if image_tensor.dim() == 3:
#         image_numpy = image_tensor.cpu().float().numpy()
#     else:
#         image_numpy = image_tensor[0].cpu().float().numpy()
#     image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * bytes

#     return image_numpy.astype(imtype)

class RunningAverage:
    def __init__(self):
        self.avg = torch.tensor(0)
        self.count = torch.tensor(0)

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg

class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}



# class RunningAverage:
#     def __init__(self):
#         self.avg = 0
#         self.count = 0

#     def append(self, value):
#         self.avg = (value + self.count * self.avg) / (self.count + 1)
#         self.count += 1

#     def get_value(self):
#         return self.avg

# class RunningAverageDict:
#     def __init__(self):
#         self._dict = None

#     def update(self, new_dict):
#         if self._dict is None:
#             self._dict = dict()
#             for key, value in new_dict.items():
#                 self._dict[key] = RunningAverage()

#         for key, value in new_dict.items():
#             self._dict[key].append(value)

#     def get_value(self):
#         return {key: value.get_value() for key, value in self._dict.items()}


limb_mask_indices_ue = [[2,4,6],
                        [3,5,7],
                        [8,10,12],
                        [9,11,13]]

limb_mask_indices_egocap = [[2,3,4],
                            [6,7,8],
                            [10,11,12],
                            [14,15,16]]



def get_limb_mask_indices(joint_preset="UnrealEgo"):
    if joint_preset == "UnrealEgo":
        return limb_mask_indices_ue
    if joint_preset == "EgoCap":
        return limb_mask_indices_egocap

# For EgoGlass
def generate_pseudo_limb_mask(pts2d, res=256, thickness=30, joint_preset="UnrealEgo"):
    thickness = 30
    thickness = thickness * res // 256
    limb_mask_indices = get_limb_mask_indices(joint_preset)
    mask = np.zeros((len(limb_mask_indices), res, res))
    pose = pts2d * res / 1024

    for i, limb in enumerate(limb_mask_indices):
        for parent, child in zip(limb[:-1], limb[1:]):
            parent_pose = tuple(map(int, pose[parent]))
            child_pose = tuple(map(int, pose[child]))
            color = 255  # White color in grayscale
            cv2.line(mask[i], tuple(parent_pose), tuple(child_pose), color, thickness)

    # Convert to binary mask
    binary_mask = (mask > 0).astype(np.float32)

    return binary_mask




# Some functions are borrowed from https://github.com/akanazawa/human_dynamics/blob/master/src/evaluation/eval_util.py
# Adhere to their licence to use these functions

def compute_accel(joints):
    """
    Computes acceleration of 3D joints.
    Args:
        joints (Nx25x3).
    Returns:
        Accelerations (N-2).
    """
    velocities = joints[1:] - joints[:-1]
    acceleration = velocities[1:] - velocities[:-1]
    acceleration_normed = np.linalg.norm(acceleration, axis=2)
    return np.mean(acceleration_normed, axis=1)


def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)


def compute_error_verts(pred_verts, target_verts=None, target_theta=None):
    """
    Computes MPJPE over 6890 surface vertices.
    Args:
        verts_gt (Nx6890x3).
        verts_pred (Nx6890x3).
    Returns:
        error_verts (N).
    """

    if target_verts is None:
        from lib.models.smpl import SMPL_MODEL_DIR
        from lib.models.smpl import SMPL
        device = 'cpu'
        smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=1, # target_theta.shape[0],
        ).to(device)

        betas = torch.from_numpy(target_theta[:,75:]).to(device)
        pose = torch.from_numpy(target_theta[:,3:75]).to(device)

        target_verts = []
        b_ = torch.split(betas, 5000)
        p_ = torch.split(pose, 5000)

        for b,p in zip(b_,p_):
            output = smpl(betas=b, body_pose=p[:, 3:], global_orient=p[:, :3], pose2rot=True)
            target_verts.append(output.vertices.detach().cpu().numpy())

        target_verts = np.concatenate(target_verts, axis=0)

    assert len(pred_verts) == len(target_verts)
    error_per_vert = np.sqrt(np.sum((target_verts - pred_verts) ** 2, axis=2))
    return np.mean(error_per_vert, axis=1)


def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # print('X1', X1.shape)

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1 ** 2)

    # print('var', var1.shape)

    # 3. The outer product of X1 and X2.
    K = X1.mm(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)
    # V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[0], device=S1.device)
    Z[-1, -1] *= torch.sign(torch.det(U @ V.T))
    # Construct R.
    R = V.mm(Z.mm(U.T))

    # print('R', X1.shape)

    # 5. Recover scale.
    scale = torch.trace(R.mm(K)) / var1
    # print(R.shape, mu1.shape)
    # 6. Recover translation.
    t = mu2 - scale * (R.mm(mu1))
    # print(t.shape)

    # 7. Error:
    S1_hat = scale * R.mm(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def batch_compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0,2,1)

    return S1_hat


def align_by_pelvis(joints):
    """
    Assumes joints is 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """

    left_id = 2
    right_id = 3

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.0
    return joints - np.expand_dims(pelvis, axis=0)


def compute_errors(gt3ds, preds):
    """
    Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
    Evaluates on the 14 common joints.
    Inputs:
      - gt3ds: N x 14 x 3
      - preds: N x 14 x 3
    """
    errors, errors_pa = [], []
    for i, (gt3d, pred) in enumerate(zip(gt3ds, preds)):
        gt3d = gt3d.reshape(-1, 3)
        # Root align.
        gt3d = align_by_pelvis(gt3d)
        pred3d = align_by_pelvis(pred)

        joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1))
        errors.append(np.mean(joint_error))

        # Get PA error.
        pred3d_sym = compute_similarity_transform(pred3d, gt3d)
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym)**2, axis=1))
        errors_pa.append(np.mean(pa_error))

    return errors, errors_pa



########################################################################################################

def get_feat_patches(x, kc, kh, kw, dc, dh, dw):
    x = x.unsqueeze(0)  # (256, 24, 32) to (1, 256, 24, 32)
    patches = x.unfold(1, int(kc), int(dc)).unfold(2, int(kh), int(dh)).unfold(3, int(kw), int(dw))
    patches = patches.contiguous().view(-1, int(kc), int(kh), int(kw))

    return patches

    # tensor_vec = get_feat_patches( # (256, 24, 32) to (1, 256, 24, 32) to (4, 256, 12, 16)
    #     tensor,
    #     kc=self.z_numchnls, kh=self.z_height/self.split_scale, kw=self.z_width/self.split_scale,
    #     dc=1, dh=self.z_height/self.split_scale, dw=self.z_width/self.split_scale
    #     )
    # tensor_vec = tensor_vec.view(-1,int(self.z_numchnls*self.z_height*self.z_width/self.split_scale/self.split_scale))  # z tensor to a vector (4, 256*12*16)

    # size = patches.size()[0]
    # imgid_new = []
    # for i in range(len(imgid)): # batch size
    #     for t in range(int(size/len(imgid))):
    #         imgid_new.append(imgid[i] + '_{}'.format(t))

    # assert size == len(imgid_new)


########################################################################################################


def trans_qrot_to_matrix(trans, rot):
    # rot is quat
    rot_mat = Rotation.from_quat(rot).as_matrix()
    mat = np.array([
        np.concatenate([rot_mat[0], [trans[0]]]),
        np.concatenate([rot_mat[1], [trans[1]]]),
        np.concatenate([rot_mat[2], [trans[2]]]),
        [0, 0, 0, 1]
    ])
    return mat

def transformation_matrix_to_translation_and_rotation(mat):
    rotation_matrix = mat[:3, :3]
    translation = mat[:3, 3]

    # rotation matrix to rotation euler
    rotation_euler = Rotation.from_matrix(rotation_matrix).as_euler('xyz')
    return rotation_euler, translation

def local_skeleton_2_global_skeleton(local_pose_list, cam_2_world_mat):
    pass

def global_skeleton_2_local_skeleton(global_pose, world_2_cam_mat):
    global_pose_homo = np.concatenate([global_pose, np.ones(1)], axis=1)
    local_pose_homo = world_2_cam_mat.dot(global_pose_homo.T).T
    return local_pose_homo

def transform_pose(pose, matrix):
    pose_homo = np.concatenate([pose, np.ones(shape=(pose.shape[0], 1))], axis=1)
    new_pose_homo = matrix.dot(pose_homo.T).T
    new_pose = new_pose_homo[:, :3]
    return new_pose

def transform_pose_torch(pose, matrix):
    to_attach = torch.ones(size=(pose.shape[0], 1)).to(pose.device)
    pose_homo = torch.cat([pose, to_attach], dim=1)
    new_pose_homo = matrix.mm(pose_homo.T).T
    new_pose = new_pose_homo[:, :3]
    return new_pose

def get_concecutive_global_cam(cam_seq, last_cam):
    camera_matrix_0 = deepcopy(cam_seq[0])
    concecutive_global_cam = np.empty_like(cam_seq)
    camera_matrix_0_inv = np.linalg.inv(camera_matrix_0)
    for i, camera_pose_i in enumerate(cam_seq):
        camera_matrix_i = camera_pose_i
        camera_matrix_i_2_last = last_cam.dot(camera_matrix_0_inv.dot(camera_matrix_i))
        concecutive_global_cam[i] = camera_matrix_i_2_last
    return concecutive_global_cam

def get_relative_global_pose(local_pose_list, camera_pose_list):
    # firstly get relative camera pose list
    relative_pose_list = []
    camera_pose_0 = deepcopy(camera_pose_list[0])
    camera_matrix_0 = trans_qrot_to_matrix(camera_pose_0['loc'], camera_pose_0['rot'])
    camera_matrix_0_inv = np.linalg.inv(camera_matrix_0)
    for i, camera_pose_i in enumerate(camera_pose_list):
        camera_matrix_i = trans_qrot_to_matrix(camera_pose_i['loc'], camera_pose_i['rot'])
        camera_matrix_i_2_0 = camera_matrix_0_inv.dot(camera_matrix_i)
        local_pose = local_pose_list[i]
        transformed_local_pose = transform_pose(local_pose, camera_matrix_i_2_0)
        relative_pose_list.append(transformed_local_pose)
    return relative_pose_list

def get_relative_global_pose_with_camera_matrix(local_pose_list, camera_pose_list):
    # firstly get relative camera pose list
    relative_pose_list = []
    camera_pose_0 = deepcopy(camera_pose_list[0])
    camera_matrix_0 = camera_pose_0
    camera_matrix_0_inv = np.linalg.inv(camera_matrix_0)
    for i, camera_pose_i in enumerate(camera_pose_list):
        camera_matrix_i = camera_pose_i
        camera_matrix_i_2_0 = camera_matrix_0_inv.dot(camera_matrix_i)
        local_pose = local_pose_list[i]
        transformed_local_pose = transform_pose(local_pose, camera_matrix_i_2_0)
        relative_pose_list.append(transformed_local_pose)

    return np.asarray(relative_pose_list)

def get_global_pose_from_relative_global_pose(relative_global_pose_list, initial_camera_matrix):
    global_pose_list = np.zeros_like(relative_global_pose_list)
    for i, relative_global_pose in enumerate(relative_global_pose_list):
        global_pose = transform_pose(relative_global_pose, initial_camera_matrix)
        global_pose_list[i] = global_pose
    return global_pose_list

def get_relative_camera_matrix(camera_pose_1, camera_pose_2):
    camera_matrix_1_inv = torch.inverse(camera_pose_1)
    camera_matrix_2_to_1 = camera_matrix_1_inv @ camera_pose_2
    return camera_matrix_2_to_1

def get_relative_global_pose_with_camera_matrix_torch(local_pose_list, camera_pose_list):
    # firstly get relative camera pose list
    # relative_pose_list = []
    relative_pose_list = torch.zeros_like(local_pose_list)
    camera_pose_0 = camera_pose_list[0].detach().clone()
    camera_matrix_0 = camera_pose_0
    camera_matrix_0_inv = torch.inverse(camera_matrix_0)
    for i, camera_pose_i in enumerate(camera_pose_list):
        camera_matrix_i = camera_pose_i
        camera_matrix_i_2_0 = camera_matrix_0_inv.mm(camera_matrix_i)
        local_pose = local_pose_list[i]
        transformed_local_pose = transform_pose_torch(local_pose, camera_matrix_i_2_0)
        relative_pose_list[i] = transformed_local_pose
    return relative_pose_list


def get_relative_transform(location1, rotation1, location2, rotation2):
    # input: location and rotation in blender coordinate
    # out: rotation, translation and transform matrix in OpenCV coordinate
    T_world2cv1, R_world2cv1, mat_world2cv1 = get_cv_rt_from_blender(location1, rotation1)
    T_world2cv2, R_world2cv2, mat_world2cv2 = get_cv_rt_from_blender(location2, rotation2)

    mat_cv1_2world = np.linalg.inv(mat_world2cv1)

    mat_cv1_to_cv2 = mat_cv1_2world.dot(mat_world2cv2)
    # mat cv1 to cv2 is the coordinate transformation, we need to change it to the object transformation
    mat_cv2_to_cv1 = np.linalg.inv(mat_cv1_to_cv2)

    rotation, translation = transformation_matrix_to_translation_and_rotation(mat_cv2_to_cv1)
    return rotation, translation, mat_cv2_to_cv1


def get_transform_relative_to_base_cv(base_location, base_rotation, location, rotation):
    # input: location and rotation in blender coordinate
    # out: rotation, translation and transform matrix in OpenCV coordinate
    T_world2cv_base, R_world2cv_base, mat_world2cv_base = get_cv_rt_from_cv(base_location, base_rotation)
    T_world2cv2, R_world2cv2, mat_world2cv2 = get_cv_rt_from_cv(location, rotation)

    # mat_cv2world2 = np.linalg.inv(mat_world2cv2)
    # location_cv_homo = mat_cv2world2[:, 3]

    location_cv_homo = np.concatenate([location, np.ones(1)])

    R_cv2_2_base = R_world2cv2.T.dot(R_world2cv_base)
    new_rotation_euler = Rotation.from_matrix(R_cv2_2_base).as_euler(seq='xyz')

    new_location_homo = mat_world2cv_base.dot(location_cv_homo)
    new_location = new_location_homo[:3]

    return new_location, new_rotation_euler

def get_transform_relative_to_base_blender(base_location, base_rotation, location, rotation):
    T_world2cv_base, R_world2cv_base, mat_world2cv_base = get_cv_rt_from_blender(base_location, base_rotation)
    T_world2cv2, R_world2cv2, mat_world2cv2 = get_cv_rt_from_blender(location, rotation)

    location_cv_homo = np.concatenate([location, np.ones(1)])

    R_cv2_2_base = R_world2cv2.T.dot(R_world2cv_base)
    new_rotation_euler = Rotation.from_matrix(R_cv2_2_base).as_euler(seq='xyz')

    new_location_homo = mat_world2cv_base.dot(location_cv_homo)
    new_location = new_location_homo[:3]

    return new_location, new_rotation_euler

# code modified from zaw lin
def get_cv_rt_from_blender(location, rotation):
    # bcam stands for blender camera
    R_bcam2cv = np.array(
        [[1, 0, 0],
         [0, -1, 0],
         [0, 0, -1]])

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints

    R_world2bcam = Rotation.from_euler('xyz', rotation, degrees=False).as_matrix().T

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam.dot(location)

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv.dot(R_world2bcam)
    T_world2cv = R_bcam2cv.dot(T_world2bcam)

    #put into 3x4 matrix
    mat = np.array([
        np.concatenate([R_world2cv[0], [T_world2cv[0]]]),
        np.concatenate([R_world2cv[1], [T_world2cv[1]]]),
        np.concatenate([R_world2cv[2], [T_world2cv[2]]]),
        [0, 0, 0, 1]
    ])
    return T_world2cv, R_world2cv, mat

# code modified from zaw lin
def get_cv_rt_from_cv(location, rotation):

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints

    R_world2cv = Rotation.from_euler('xyz', rotation, degrees=False).as_matrix().T

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2cv = -1 * R_world2cv.dot(location)

    #put into 3x4 matrix
    mat = np.array([
        np.concatenate([R_world2cv[0], [T_world2cv[0]]]),
        np.concatenate([R_world2cv[1], [T_world2cv[1]]]),
        np.concatenate([R_world2cv[2], [T_world2cv[2]]]),
        [0, 0, 0, 1]
    ])
    return T_world2cv, R_world2cv, mat