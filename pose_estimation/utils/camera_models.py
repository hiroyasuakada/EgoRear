# Author: Hiroyasu Akada

import os
import torch
import json

from einops import rearrange
from torch._dynamo import allow_in_graph
allow_in_graph(rearrange)

from loguru import logger


class FishEyeCameraCalibratedModel:
    def __init__(self, camera_model, camera_calib_file_dir_path, camera_name):

        self.camera_model = camera_model
        self.camera_name = camera_name

        camera_calib_file_path = os.path.join(camera_calib_file_dir_path, "{}.json".format(camera_name))
        with open(camera_calib_file_path) as f:
            calibration_data = json.load(f)

        self.image_size = torch.tensor(calibration_data['size'], device="cuda")
        self.image_center = torch.tensor(calibration_data['image_center'], device="cuda")
        self.fisheye_polynomial = torch.tensor(calibration_data['polynomialC2W'], device="cuda")
        self.fisheye_inv_polynomial = torch.tensor(calibration_data['polynomialW2C'], device="cuda")

        if camera_model.startswith("ego4view_syn"):
            # avoid matrix computation for faster training and inference
            if camera_name == "camera_front_left":
                offset = [6.0, 0.0, 0.0]
            elif camera_name == "camera_front_right":
                offset = [-6.0, 0.0, 0.0]
            elif camera_name == "camera_back_left":
                offset = [-6.0, 37.0, 0.0]
            elif camera_name == "camera_back_right":
                offset = [6.0, 37.0, 0.0]

            self.offset = torch.tensor(offset, device="cuda")

        elif (camera_model.startswith("ego4view_rw")):
            logger.info("")
            logger.info("We will use coordinate transformation matrix to convert device-relative pose to camera-relative pose")
            logger.info("")

        else:
            raise ValueError('Unknown camera model !')

        self.m2cm = 100.0
        self.cm2m = 0.01

    def get_camera_relative_pts3d(self, pts3d, coord_trans_mat=None):
        if (self.camera_model.startswith("ego4view_rw")):
            pts3d = apply_batch_transformation_matrix_pytorch(pts3d * self.cm2m, coord_trans_mat) * self.m2cm

        elif self.camera_model.startswith("ego4view_syn"):

            if self.camera_name == "camera_back_left":
                pts3d[..., 0:2] *= -1
            elif self.camera_name == "camera_back_right":
                pts3d[..., 0:2] *= -1
            pts3d += self.offset

        else:
            raise ValueError('Unknown camera model !')

        return pts3d

    def world2camera_pytorch(self, pts3d_original, coord_trans_mat=None):

        with torch.no_grad():

            pts3d = self.get_camera_relative_pts3d(pts3d_original, coord_trans_mat)

            pts3d = pts3d[:, None].repeat(1, 1, 1, 1)

            x = pts3d[..., 0]
            y = pts3d[..., 1]
            z = pts3d[..., 2]

            norm = torch.sqrt(x * x + y * y)
            theta = torch.atan(-z / norm)

            rho = sum(a * theta ** i for i, a in enumerate(self.fisheye_inv_polynomial))

            u = x / norm * rho + self.image_center[0]
            v = y / norm * rho + self.image_center[1]

            u = u / self.image_size[1]
            v = v / self.image_size[0]

            point2d = torch.stack((u, v), dim=-1)

            in_fov = (
                    (point2d[..., 0] > 0)
                    & (point2d[..., 1] > 0)
                    & (point2d[..., 0] < 1)
                    & (point2d[..., 1] < 1)
            )

            point2d = point2d.clamp(min=0.0, max=1.0)

            return point2d, in_fov

def unrealego_proj(local_3d, local_origin=None):
    num_views = 2
    polynomial_w2c = (
        541.084422, 133.996745, -53.833198, 60.96083, -24.78051, 12.451492,
        -30.240511, 26.90122, 116.38499, -133.991117, -141.904687, 184.05592,
        107.45616, -125.552875, -55.66342, 44.209519, 18.234651, -6.410899, -2.737066
    )
    image_center = (511.1183388444314, 510.8730105600536)
    raw_image_size = (1024, 1024)

    with torch.no_grad():
        cam_3d = local_3d[:, None].repeat(1, num_views, 1, 1)

        if isinstance(local_origin, torch.Tensor):
            cam_3d = cam_3d + local_origin
        elif local_origin == None:
            cam_3d[:, 0, :, 0] += -6.0
            cam_3d[:, 1, :, 0] += 6.0

        else:
            raise NotImplementedError

        x = cam_3d[..., 0]
        y = cam_3d[..., 1]
        z = cam_3d[..., 2]

        norm = torch.sqrt(x * x + y * y)
        theta = torch.atan(-z / norm)

        rho = sum(a * theta ** i for i, a in enumerate(polynomial_w2c))

        u = x / norm * rho + image_center[0]
        v = y / norm * rho + image_center[1]

        u = u / raw_image_size[1]
        v = v / raw_image_size[0]

        image_coor_2d = torch.stack((u, v), dim=-1)
        in_fov = (
                (image_coor_2d[..., 0] > 0)
                & (image_coor_2d[..., 1] > 0)
                & (image_coor_2d[..., 0] < 1)
                & (image_coor_2d[..., 1] < 1)
        )
        image_coor_2d = image_coor_2d.clamp(min=0.0, max=1.0)

        return image_coor_2d, in_fov

projection_funcs = {
    "unrealego": unrealego_proj,
    "unrealego2": unrealego_proj
}

def apply_transformation_matrix_pytorch(pts3d, transformation_matrix):
    """
    Apply a transformation matrix to a batch of 3D points.

    Args:
        pts3d (torch.Tensor): A tensor of shape (N, 3) where N is the number of 3D points.
        transformation_matrix (torch.Tensor): A tensor of shape (4, 4) representing the transformation matrix.

    Returns:
        torch.Tensor: A tensor of transformed 3D points of shape (N, 3).
    """
    # Get the number of 3D points
    N = pts3d.size()[0]

    # Convert 3D points to homogeneous coordinates by adding a column of ones (shape (N, 4))
    ones = torch.ones((N, 1), dtype=pts3d.dtype, device=pts3d.device)
    pts3d_homogeneous = torch.cat([pts3d, ones], dim=1)  # Shape: (N, 4)

    # Apply the transformation matrix (matmul will apply the matrix to each point)
    transformed_pts3d_homogeneous = torch.matmul(transformation_matrix, pts3d_homogeneous.T).T # Shape: (N, 4)

    # Extract the first three components to get back to (N, 3)
    transformed_pts3d = transformed_pts3d_homogeneous[:, :3]

    return transformed_pts3d



def apply_batch_transformation_matrix_pytorch(pts3d, transformation_matrices):
    """
    Apply a batch of transformation matrices to a batch of 3D points with joints.

    Args:
        pts3d (torch.Tensor): A tensor of shape (B, J, 3) where B is the batch size and J is the number of joints.
        transformation_matrices (torch.Tensor): A tensor of shape (B, 4, 4), with one transformation matrix per batch.

    Returns:
        torch.Tensor: A tensor of transformed 3D points of shape (B, J, 3).
    """
    # Get the batch size and number of joints
    B, J = pts3d.size()[0:2]

    # Convert 3D points to homogeneous coordinates by adding a column of ones (shape (B, J, 4))
    ones = torch.ones((B, J, 1), dtype=pts3d.dtype, device=pts3d.device)
    pts3d_homogeneous = torch.cat([pts3d, ones], dim=2)  # Shape: (B, J, 4)

    # Expand transformation_matrices to match the joint dimension (B, 1, 4, 4) -> (B, J, 4, 4)
    transformation_matrices = transformation_matrices.unsqueeze(1).expand(-1, J, -1, -1)

    # Apply the transformation matrix to each point using batch matrix multiplication
    # (B, J, 4, 4) x (B, J, 4, 1) -> (B, J, 4)
    transformed_pts3d_homogeneous = torch.matmul(transformation_matrices, pts3d_homogeneous.unsqueeze(3)).squeeze(3)  # Shape: (B, J, 4)

    # Extract the first three components to get back to (B, J, 3)
    transformed_pts3d = transformed_pts3d_homogeneous[:, :, :3]

    return transformed_pts3d
