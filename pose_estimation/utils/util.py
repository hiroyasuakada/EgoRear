# Author: Hiroyasu Akada

import numpy as np
import torch
import torch.nn as nn
import os
from PIL import Image
import open3d
from scipy.spatial.transform import Rotation
from copy import deepcopy

from pose_estimation.utils.skeleton import Skeleton
skeleton_model = Skeleton(calibration_path=None)

def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8, is_depth=False, is_heatmap=False, is_video=False, is_indi_heatmap=False):

    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0].cpu().float()
    elif image_tensor.dim() == 3:
        image_tensor = image_tensor.cpu().float()
    elif image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0).cpu().float()
    else:
        raise NotImplementedError

    if is_video:
        image_tensor = image_tensor[-1]

    if is_depth:
        image_tensor = image_tensor * bytes
    elif is_heatmap:
        image_tensor = torch.clamp(torch.sum(image_tensor, dim=0, keepdim=True), min=0.0, max=1.0) * bytes
    else:
        image_tensor = denormalize_ImageNet(image_tensor) * bytes

    image_numpy = (image_tensor.permute(1, 2, 0)).detach().numpy().astype(imtype)
    return image_numpy

def tensor2pose(joints_3d, is_video=False):
    joints_3d = joints_3d[0].cpu().float()

    if is_video:
        joints_3d = joints_3d[-1]

    return joints_3d

def tensor2meshpose(joints_3d, save_name, color):

    mesh = skeleton_model.joints_2_mesh(joints_3d, color)

    save_dir = os.path.join(os.path.dirname(save_name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    open3d.io.write_triangle_mesh(save_name, mesh)

    return mesh

def tensor2meshpose_colorful(joints_3d, save_name):

    mesh = skeleton_model.joints_2_mesh(joints_3d, colorful_pose=True)

    save_dir = os.path.join(os.path.dirname(save_name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    open3d.io.write_triangle_mesh(save_name, mesh)

    return mesh

def apply_softmax(heatmap):
    H, W = heatmap.shape
    y_indices, x_indices = np.indices((H, W))
    normalized_heatmap = np.exp(heatmap) / np.sum(np.exp(heatmap))
    y = np.sum(y_indices * normalized_heatmap)
    x = np.sum(x_indices * normalized_heatmap)
    return x, y


def integrate_tensor_2d(heatmaps, softmax=True, multiplier=100.0):
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

def denormalize_ImageNet(x):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return x * std + mean

def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])
    image_numpy = Image.fromarray(image_numpy)
    image_numpy.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

def compute_similarity_transform(S1, S2):
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = np.sum(X1**2)

    K = X1.dot(X2.T)

    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    R = V.dot(Z.dot(U.T))

    scale = np.trace(R.dot(K)) / var1

    t = mu2 - scale*(R.dot(mu1))

    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_torch(S1, S2):

    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = torch.sum(X1 ** 2)

    K = X1.mm(X2.T)

    U, s, V = torch.svd(K)
    Z = torch.eye(U.shape[0], device=S1.device)
    Z[-1, -1] *= torch.sign(torch.det(U @ V.T))
    R = V.mm(Z.mm(U.T))

    scale = torch.trace(R.mm(K)) / var1

    t = mu2 - scale * (R.mm(mu1))

    S1_hat = scale * R.mm(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def batch_compute_similarity_transform_torch(S1, S2):
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    K = X1.bmm(X2.permute(0,2,1))

    U, s, V = torch.svd(K)

    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0,2,1)

    return S1_hat


def align_by_pelvis(joints):
    left_id = 2
    right_id = 3

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.0
    return joints - np.expand_dims(pelvis, axis=0)


def compute_errors(gt3ds, preds):
    errors, errors_pa = [], []
    for i, (gt3d, pred) in enumerate(zip(gt3ds, preds)):
        gt3d = gt3d.reshape(-1, 3)
        gt3d = align_by_pelvis(gt3d)
        pred3d = align_by_pelvis(pred)

        joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1))
        errors.append(np.mean(joint_error))

        pred3d_sym = compute_similarity_transform(pred3d, gt3d)
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym)**2, axis=1))
        errors_pa.append(np.mean(pa_error))

    return errors, errors_pa

def get_feat_patches(x, kc, kh, kw, dc, dh, dw):
    x = x.unsqueeze(0)
    patches = x.unfold(1, int(kc), int(dc)).unfold(2, int(kh), int(dh)).unfold(3, int(kw), int(dw))
    patches = patches.contiguous().view(-1, int(kc), int(kh), int(kw))

    return patches

def trans_qrot_to_matrix(trans, rot):
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
    T_world2cv1, R_world2cv1, mat_world2cv1 = get_cv_rt_from_blender(location1, rotation1)
    T_world2cv2, R_world2cv2, mat_world2cv2 = get_cv_rt_from_blender(location2, rotation2)

    mat_cv1_2world = np.linalg.inv(mat_world2cv1)
    mat_cv1_to_cv2 = mat_cv1_2world.dot(mat_world2cv2)
    mat_cv2_to_cv1 = np.linalg.inv(mat_cv1_to_cv2)

    rotation, translation = transformation_matrix_to_translation_and_rotation(mat_cv2_to_cv1)
    return rotation, translation, mat_cv2_to_cv1


def get_transform_relative_to_base_cv(base_location, base_rotation, location, rotation):
    T_world2cv_base, R_world2cv_base, mat_world2cv_base = get_cv_rt_from_cv(base_location, base_rotation)
    T_world2cv2, R_world2cv2, mat_world2cv2 = get_cv_rt_from_cv(location, rotation)

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

def get_cv_rt_from_blender(location, rotation):
    R_bcam2cv = np.array(
        [[1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]])


    R_world2bcam = Rotation.from_euler('xyz', rotation, degrees=False).as_matrix().T
    T_world2bcam = -1 * R_world2bcam.dot(location)

    R_world2cv = R_bcam2cv.dot(R_world2bcam)
    T_world2cv = R_bcam2cv.dot(T_world2bcam)

    mat = np.array([
        np.concatenate([R_world2cv[0], [T_world2cv[0]]]),
        np.concatenate([R_world2cv[1], [T_world2cv[1]]]),
        np.concatenate([R_world2cv[2], [T_world2cv[2]]]),
        [0, 0, 0, 1]
    ])
    return T_world2cv, R_world2cv, mat

def get_cv_rt_from_cv(location, rotation):
    R_world2cv = Rotation.from_euler('xyz', rotation, degrees=False).as_matrix().T
    T_world2cv = -1 * R_world2cv.dot(location)

    mat = np.array([
        np.concatenate([R_world2cv[0], [T_world2cv[0]]]),
        np.concatenate([R_world2cv[1], [T_world2cv[1]]]),
        np.concatenate([R_world2cv[2], [T_world2cv[2]]]),
        [0, 0, 0, 1]
    ])
    return T_world2cv, R_world2cv, mat