import os
import glob
import cv2
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
import sys
import torch
from pytorch3d.transforms import axis_angle_to_matrix

sys.path.append(".")
sys.path.append("../")

from utils.cam_utils import get_camera_params, fov2K, get_camera_params_360, cv2gl


def load_mdm_pose(path):
    data = dict(np.load(path, allow_pickle=True).item())
    poses = np.transpose(data["thetas_ori"].cpu().numpy(), (2, 0, 1))
    Rh = poses[:, 0].copy()
    Th = np.transpose(data["root_translation"], (1, 0))
    poses[:, 0] = 0.0
    poses = poses.reshape(poses.shape[0], -1)
    pose_infos = {
        "body_pose": poses,
        "global_orient": Rh,
        "transl": Th,
    }
    return pose_infos


def load_pose(pose_path):
    """return (num_frame, 72)"""
    poses = np.load(pose_path, allow_pickle=True)
    return poses


class AnimateDataset(Dataset):

    def __init__(
        self,
        width=512,
        height=512,
        bgcolor=[1.0, 1.0, 1.0],
        pose_path=None,
        pose=None,
        is360=False,
    ):
        """
        pose: (N, 72) or None
        """
        super().__init__()

        if pose is not None:
            global_orient = torch.zeros((len(pose), 3), dtype=torch.float32)
            self.full_poses = torch.cat([global_orient, pose], dim=1)
        elif pose_path is not None:
            self.full_poses = torch.tensor(load_pose(pose_path), dtype=torch.float32)
            # self.full_poses[:, :3] = 0.0  # set global_orient to zero
        else:
            raise ValueError("Either pose or pose_path should be provided.")

        self.is360 = is360

        if bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.0).astype(np.float32)
        else:
            bgcolor = np.array(bgcolor, dtype=np.float32)
        self.bgcolor = torch.tensor(bgcolor / 255.0, dtype=torch.float32)

        self.transl = torch.tensor([0, 0, 0], dtype=torch.float32)
        self.full_poses = self.full_poses

        if is360:
            self.camera_params = get_camera_params_360(width, height, num_deg=len(self))
        else:
            c2w = torch.eye(4)
            c2w[:3, 3] = torch.tensor([0, 0, -2.5], dtype=torch.float32)
            intrinsic = fov2K(60, H=height, W=width).astype(np.float32)
            # extrinsic = torch.tensor(cv2gl) @ torch.inverse(c2w)
            extrinsic = torch.inverse(c2w)
            self.camera_params = get_camera_params(intrinsic, extrinsic, width, height)

    def __len__(self):
        return len(self.full_poses)

    def __getitem__(self, idx):

        smpl_params = {
            "full_pose": self.full_poses[idx],  # (N, 72)
            "transl": self.transl,
        }
        camera_param = self.camera_params[idx] if self.is360 else self.camera_params
        ret = {
            "idx": idx,
            "bgcolor": self.bgcolor,
            "smpl_params": smpl_params,
            "camera_params": camera_param,
        }

        return ret


if __name__ == "__main__":

    dataset = AnimateDataset(
        pose_path="./novel_poses/walking.npy", bgcolor=[255, 255, 255]
    )
    for i in range(len(dataset)):
        print(f"Item {i}:")
        print(dataset[i])
        print("-" * 20)
