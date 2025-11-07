import sys
import os

sys.path.append(os.getcwd())
import numpy as np
import os.path as osp
import numpy as np
import cv2
import glob
import torch

# from utils.cam_utils import get_camera_params
from torch.utils.data import Dataset
from tqdm import tqdm
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
from pytorch3d.structures import Meshes
import logging
import pickle
from utils.cam_utils import get_camera_params


def img2mask(img):
    mask = np.where(np.all(img == [255, 255, 255], axis=-1), 0, 1)
    return mask.astype(np.uint8)


def sort_rgb(x):
    return int(os.path.basename(x).split(".")[0].split("_")[-1])


def sort_smpl(x):
    return int(os.path.basename(x).split(".")[0].split("_")[0].split("-")[-1][1:])


def sort_mesh(x):
    name = os.path.basename(x)
    return int(name.split(".")[0].split("f")[-1])


def load_smpl_params(smpl_path, smpl_type="smpl"):

    # mesh-f00002_smplx.pkl
    smpl_lists = sorted(glob.glob(f"{smpl_path}/*.pkl"), key=sort_smpl)
    smpl_params = []
    for smpl_file in smpl_lists:
        smpl_param = np.load(smpl_file, allow_pickle=True)
        for key in smpl_param.keys():
            smpl_param[key] = torch.tensor(smpl_param[key])

        smpl_params.append(smpl_param)

    return smpl_params


class XhumanDataset(Dataset):
    def __init__(
        self,
        data_root="./data",
        subject="00019",
        x_split="train",
        split="train",
        take="Take1",
        data_split=None,
        smpl_type="SMPLX",
        image_zoom_ratio=0.5,
        cfg=None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.subject = subject
        self.image_zoom_ratio = image_zoom_ratio
        self.random_bg = split == "train" and cfg.get("random_bg", False)
        self.bgcolor = cfg.get("bgcolor", [1.0, 1.0, 1.0])
        self.load_depth = cfg.get("load_depth", False)

        start = data_split.start
        end = data_split.end
        skip = data_split.skip

        self.smpl_type = smpl_type

        subject_root = osp.join(data_root, subject)
        root = osp.join(data_root, subject, x_split, take)
        self.root = root

        self.label = f"{subject}_{x_split}_{take}"

        self.downscale = 1.0 / self.image_zoom_ratio

        image_path = osp.join(root, "render", "image")
        normal_path = osp.join(root, "sapiens", "normals")

        self.img_lists = sorted(glob.glob(f"{image_path}/*.png"), key=sort_rgb)[
            start:end:skip
        ]
        self.normal_lists = sorted(glob.glob(f"{normal_path}/*.png"), key=sort_rgb)[
            start:end:skip
        ]

        if self.load_depth:
            depth_path = osp.join(root, "sapiens", "depths")
            self.depth_lists = sorted(glob.glob(f"{depth_path}/*.png"), key=sort_rgb)[
                start:end:skip
            ]

        with open(osp.join(subject_root, "gender.txt"), "r") as f:
            self.gender = f.read().strip()

        smpl_path = osp.join(root, smpl_type.upper())
        self.smpl_params = load_smpl_params(smpl_path)[start:end:skip]

        # load camera
        camera = np.load(osp.join(root, "render", "cameras.npz"))
        self.cam_intrinsic = camera["intrinsic"].astype(np.float32)
        self.cam_extrinsic = camera["extrinsic"].astype(np.float32)[start:end:skip]
        self.img_height = camera["intrinsic"][1, 2] * 2
        self.img_width = camera["intrinsic"][0, 2] * 2
        # cat full pose
        if self.smpl_type == "smpl":
            for smpl_param in self.smpl_params:
                smpl_param["full_pose"] = torch.cat(
                    [smpl_param["global_orient"], smpl_param["body_pose"]], dim=0
                )
        elif self.smpl_type == "smplx":
            for smpl_param in self.smpl_params:
                smpl_param["full_pose"] = torch.cat(
                    [
                        smpl_param["global_orient"],
                        smpl_param["body_pose"],
                        smpl_param["jaw_pose"],
                        smpl_param["leye_pose"],
                        smpl_param["reye_pose"],
                        smpl_param["left_hand_pose"],
                        smpl_param["right_hand_pose"],  #
                    ],
                    dim=0,
                )

        if self.downscale > 1:
            self.cam_intrinsic = self.cam_intrinsic / self.downscale
            self.img_height = int(self.img_height / self.downscale)
            self.img_width = int(self.img_width / self.downscale)

        self.camera_params = []
        for extrinsic in self.cam_extrinsic:
            cam_param = get_camera_params(
                intrinsic=self.cam_intrinsic,
                extrinsic=torch.tensor(extrinsic),
                height=self.img_height,
                width=self.img_width,
            )
            self.camera_params.append(cam_param)

        self.scan_meshes = self._load_scan_mesh(start, end, skip)

        # cache the images
        self.img_buffer = []
        self.msk_buffer = []
        self.bg_buffer = []
        self.normal_buffer = []
        self.depth_buffer = []
        self.img_names = []

        for idx in tqdm(
            range(len(self.img_lists)), desc=f"Loading {self.label} images"
        ):

            img_name = os.path.basename(self.img_lists[idx]).split(".")[0]
            self.img_names.append(img_name)

            img = cv2.imread(self.img_lists[idx])[..., ::-1]
            normal = cv2.imread(self.normal_lists[idx])[..., ::-1]
            msk = img2mask(img)

            if self.load_depth:
                depth = cv2.cvtColor(
                    cv2.imread(self.depth_lists[idx]), cv2.COLOR_BGR2GRAY
                )

            if self.downscale > 1:
                img = cv2.resize(
                    img, dsize=None, fx=1 / self.downscale, fy=1 / self.downscale
                )
                msk = cv2.resize(
                    msk, dsize=None, fx=1 / self.downscale, fy=1 / self.downscale
                )
                normal = cv2.resize(
                    normal, dsize=None, fx=1 / self.downscale, fy=1 / self.downscale
                )

                if self.load_depth:
                    depth = cv2.resize(
                        depth, dsize=None, fx=1 / self.downscale, fy=1 / self.downscale
                    )

            img = (img[..., :3] / 255).astype(np.float32)
            msk = msk.astype(np.float32)
            normal = (normal[..., :3] / 255).astype(np.float32)

            if self.load_depth:
                depth = (depth / 255).astype(np.float32)
                m = msk > 0
                depth[m] = (depth[m] - depth[m].min()) / (
                    depth[m].max() - depth[m].min()
                )
                depth[~m] = 0.0

            if self.random_bg:
                bgcolor = (np.random.rand(3)).astype(np.float32)
            else:
                bgcolor = np.array(self.bgcolor, dtype=np.float32)

            img = img * msk[..., None] + (1 - msk[..., None]) * bgcolor[None, None, :]
            normal = (
                normal * msk[..., None] + (1 - msk[..., None]) * bgcolor[None, None, :]
            )

            img = torch.tensor(img).permute(2, 0, 1)
            normal = torch.tensor(normal).permute(2, 0, 1)
            msk = torch.tensor(msk)

            self.bg_buffer.append(bgcolor)
            self.img_buffer.append(img)
            self.msk_buffer.append(msk)
            self.normal_buffer.append(normal)

            if self.load_depth:
                depth = torch.tensor(depth[None])
                self.depth_buffer.append(depth)

    def __len__(self):
        return len(self.img_buffer)

    def __getitem__(self, idx):
        img = self.img_buffer[idx]
        msk = self.msk_buffer[idx]
        bgcolor = self.bg_buffer[idx]
        normal = self.normal_buffer[idx]

        ret = {
            "idx": idx,
            "img": img,
            "mask": msk,
            "normal": normal,
            "bgcolor": bgcolor,
            "smpl_params": self.smpl_params[idx],
            "camera_params": self.camera_params[idx],
        }

        if self.load_depth:
            depth = self.depth_buffer[idx]
            ret["depth"] = depth

        if self.scan_meshes:
            ret["scan_mesh"] = self.scan_meshes[idx]

        return ret

    def get_init_beta(self):
        return self.smpl_params[0]["betas"].unsqueeze(0)

    def get_smpl_pose(self):
        poses = [p["body_pose"] for p in self.smpl_params]
        body_pose = torch.stack(poses, dim=0)
        return body_pose

    def _load_scan_mesh(self, start, end, skip):
        pkl_dir = osp.join(self.root, "meshes_pkl")
        mesh_pkl_list = sorted(glob.glob(f"{pkl_dir}/mesh-*.pkl"), key=sort_mesh)[
            start:end:skip
        ]

        mesh_list = []
        for path in tqdm(mesh_pkl_list, desc="load meshes"):
            mesh = pickle.load(open(path, "rb"))
            # vertices = torch.from_numpy(mesh["vertices"])
            # faces = torch.from_numpy(mesh["faces"])

            # mesh = Meshes(verts=[vertices],faces=[faces])
            mesh_list.append(mesh)

        # from pytorch3d.io import IO
        # IO().save_mesh(mesh_list[0], "./xhuman_scan_mesh.obj")
        return mesh_list


class XhumanDataset_Multi(Dataset):
    def __init__(
        self,
        data_root="./data",
        subject="00019",
        data_split_list=[],
        smpl_type="smpl",
        split="train",
        image_zoom_ratio=0.5,
        cfg=None,
    ):
        super().__init__()
        self.datasets = []
        for data_split in data_split_list:
            logging.info(f"Loading dataset split {data_split}")
            takes = map(lambda i: f"Take{i}", data_split.takes)
            for take in takes:
                dataset = XhumanDataset(
                    data_root=data_root,
                    subject=subject,
                    x_split=data_split.x_split,
                    take=take,
                    split=split,
                    data_split=data_split.split,
                    smpl_type=smpl_type,
                    image_zoom_ratio=image_zoom_ratio,
                    cfg=cfg,
                )
                self.datasets.append(dataset)

        self.total_length = sum([len(dataset) for dataset in self.datasets])
        self.label = f"{subject}_{data_split.x_split}"

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        dataset_idx = 0
        total_idx = 0
        while idx >= len(self.datasets[dataset_idx]):
            idx -= len(self.datasets[dataset_idx])
            total_idx += len(self.datasets[dataset_idx])
            dataset_idx += 1
        total_idx += idx

        ret = self.datasets[dataset_idx][idx]
        ret["idx"] = total_idx

        return ret

    def get_init_beta(self):
        return self.datasets[0].get_init_beta()

    def get_smpl_pose(self):
        poses = []
        for dataset in self.datasets:
            poses.append(dataset.get_smpl_pose())
        pose = torch.cat(poses, dim=0)
        return pose


class DictObj(object):
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(key, (list, tuple)):
                setattr(
                    self, key, [DictObj(x) if isinstance(x, dict) else x for x in value]
                )
            else:
                setattr(self, key, DictObj(value) if isinstance(value, dict) else value)


if __name__ == "__main__":

    data_root = "/U_20240109_SZR_SMIL/hailin/human_avatar/dataset/InstantAvatar/xhumans"

    data_split_train = {
        "x_split": "test",
        "take": "Take1",
        "start": 0,
        "end": -1,
        "skip": 1,
    }
    data_split_test = {
        "x_split": "test",
        "take": "Take1",
        "start": 0,
        "end": -1,
        "skip": 10,
    }
    train_dataset = XhumanDataset(
        data_root=data_root,
        subject="00016",
        data_split=DictObj(data_split_train),
        smpl_type="SMPL",
    )
    # test_dataset = XhumanDataset(data_root=data_root, subject="00016", data_split=DictObj(data_split_test), smpl_type="SMPLX")
    ret = train_dataset[0]
    # ret = test_dataset[0]
    # 按顺序遍历数据集，导出成视频，包含rgb、mask、depth、normal，从左到右排列的视频grid
    import imageio

    frames = []
    for idx in tqdm(range(len(train_dataset))):
        ret = train_dataset[idx]
        img = ret["img"].permute(1, 2, 0).numpy()
        msk = ret["mask"].numpy()
        normal = ret["normal"].permute(1, 2, 0).numpy()
        depth = ret["depth"].squeeze().numpy()

        img = (img * 255).astype(np.uint8)
        normal = (normal * 255).astype(np.uint8)
        msk = np.repeat((msk * 255).astype(np.uint8)[..., None], 3, axis=-1)
        depth = np.repeat((depth * 255).astype(np.uint8)[..., None], 3, axis=-1)

        frame = np.concatenate([img, msk, normal, depth], axis=1)
        frames.append(frame)

    imageio.mimsave("xhuman.mp4", frames, fps=20)
