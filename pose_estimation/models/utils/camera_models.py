# Author: Hiroyasu Akada

import torch
import json
from einops import rearrange


class FishEyeCameraCalibrated:
    def __init__(self, calib_file_path, use_gpu=False):
        with open(calib_file_path) as f:
            calibration_data = json.load(f)
        self.img_size = calibration_data['size']
        self.img_center = calibration_data['image_center']
        self.img_center_x = self.img_center[0]
        self.img_center_y = self.img_center[1]
        self.fisheye_polynomial = calibration_data['polynomialC2W']
        self.fisheye_inv_polynomial = calibration_data['polynomialW2C']
        self.use_gpu = use_gpu

    def world2camera_pytorch(self, point3d_original: torch.Tensor):
        B = point3d_original.size()[0]
        point3d = point3d_original.clone()
        point3d[:, 2] = point3d_original[:, 2] * -1
        point3d = rearrange(point3d, "b j c -> c (b j)", b=B)

        point2d = torch.empty((2, point3d.size()[-1])).to(point3d.device)

        norm = torch.norm(point3d[:2], dim=0)

        if (norm != 0).all():
            theta = torch.atan(point3d[2] / norm)
            invnorm = 1.0 / norm
            t = theta
            rho = self.fisheye_inv_polynomial[0]
            t_i = 1.0

            for i in range(1, len(self.fisheye_inv_polynomial)):
                t_i *= t
                rho += t_i * self.fisheye_inv_polynomial[i]

            x = point3d[0] * invnorm * rho
            y = point3d[1] * invnorm * rho

            point2d[0] = x + self.img_center_x
            point2d[1] = y + self.img_center_y

        else:
            point2d[0] = self.img_center_x
            point2d[1] = self.img_center_y
            raise Exception("norm is zero!")


        point2d[0] = point2d[0] / self.img_size[0]
        point2d[1] = point2d[1] / self.img_size[1]

        in_fov = (
                (point2d[0] > 0)
                & (point2d[1] > 0)
                & (point2d[0] < 1)
                & (point2d[1] < 1)
        )
        point2d = point2d.clamp(min=0.0, max=1.0)

        point2d = rearrange(point2d, "c (b j) -> b j c", b=B)

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

        if local_origin:
            cam_3d = cam_3d + local_origin
        else:
            cam_3d[:, 0, :, 0] += -6.0
            cam_3d[:, 1, :, 0] += 6.0

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
    'unrealego': unrealego_proj
}

def get_camera_model(camera_model, camera_name):
    pass