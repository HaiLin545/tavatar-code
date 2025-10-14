import torch
import numpy as np
from dataclasses import dataclass
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.renderer import FoVOrthographicCameras, look_at_view_transform

cv2gl = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


def fov2K(fov=90, H=512, W=512):
    f = H / (2 * np.tan(fov / 2 * np.pi / 180))
    K = np.eye(3)
    K[0, 0], K[0, 2] = f, W / 2.0
    K[1, 1], K[1, 2] = f, H / 2.0
    return K


def focal2tanfov(focal, pixels):
    return pixels / (2 * focal)


def getProjectionMatrixShift(
    tanHalfFovY,
    tanHalfFovX,
    focal_x,
    focal_y,
    cx,
    cy,
    width,
    height,
    znear=0.01,
    zfar=100.0,
):
    # the origin at center of image plane
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    # shift the frame window due to the non-zero principle point offsets
    offset_x = cx - (width / 2)
    offset_x = (offset_x / focal_x) * znear
    offset_y = cy - (height / 2)
    offset_y = (offset_y / focal_y) * znear

    top = top + offset_y
    left = left + offset_x
    right = right + offset_x
    bottom = bottom + offset_y

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def get_smplx_camera_params(width, height):
    c2w = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -2.2],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )
    intrinsic = fov2K(50, height, width)
    extrinsic = torch.inverse(c2w)
    return get_camera_params(intrinsic, extrinsic, width, height)


def get_camera_params(intrinsic, extrinsic, width, height):
    focal_length_x = intrinsic[0, 0]
    focal_length_y = intrinsic[1, 1]
    tanFovY = focal2tanfov(focal_length_y, height)
    tanFovX = focal2tanfov(focal_length_x, width)
    projmatrix = getProjectionMatrixShift(
        tanFovY,
        tanFovX,
        focal_x=focal_length_x,
        focal_y=focal_length_y,
        cx=intrinsic[0, 2],
        cy=intrinsic[1, 2],
        width=width,
        height=height,
    )
    viewmatrix = extrinsic
    # breakpoint()
    camera_params = {
        "image_height": height,
        "image_width": width,
        "tanfovx": tanFovX,
        "tanfovy": tanFovY,
        "scale_modifier": 1.0,
        "viewmatrix": viewmatrix.T,
        "projmatrix": (projmatrix @ viewmatrix).T,
        "campos": torch.inverse(viewmatrix)[3, :3],
        "bg": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
    }
    return camera_params


def get_camera_params_360(width=512, height=512, num_deg=100, device="cuda"):

    intrinsic = fov2K(60, H=height, W=width).astype(np.float32)
    axis = torch.tensor([0, 1, 0], dtype=torch.float32)
    camera_position = torch.tensor([0, 0, 2.5], dtype=torch.float32)
    camera_params_360 = []
    cv_2_gl = torch.tensor(cv2gl, dtype=torch.float32)

    for i in range(num_deg):
        angle = (i * 2 * np.pi) / num_deg
        R = axis_angle_to_matrix(axis * angle)
        c2w = torch.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = R @ camera_position
        extrinsic = cv_2_gl @ torch.inverse(c2w)
        camera_param = get_camera_params(intrinsic, extrinsic, width, height)
        camera_params_360.append(camera_param)

    return camera_params_360
