import os
import torch
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.cam_utils import get_camera_params, get_smplx_camera_params
import logging
from tqdm import tqdm


class HumanDataset(Dataset):
    def __init__(
        self,
        root="data/people_snapshot",
        subject="male-3-casual",
        split="train",
        smpl_type="smpl",
        opt=None,
    ) -> None:
        super().__init__()
        self.root = os.path.join(root, subject)
        logging.info(f"dataset={root}, subject={subject}")

        self.split = split
        data_split = opt.split.get(split)
        self.start = data_split.get("start")
        self.end = data_split.get("end") + 1
        self.skip = data_split.get("skip", 1)
        self.label = subject
        self.downscale = opt.get("downscale", 1.0)
        self.smpl_type = smpl_type
        self.camera_params = self.load_camera(opt)
        self.height = self.camera_params["image_height"]
        self.width = self.camera_params["image_width"]

        self.random_bg = split == "train" and opt.get("random_bg", False)
        self.bgcolor = opt.get("bgcolor", [1.0, 1.0, 1.0])

        self.load_depth = opt.get("load_depth", True)
        self.reverse_depth = opt.get("reverse_depth", False)

        self.img_lists = self.read_images(os.path.join(self.root, "images"))
        mask_dir = os.path.join(self.root, "masks")

        normal_dir = os.path.join(self.root, "sapiens", "normals")
        self.msk_lists = self.read_masks(mask_dir)
        self.normal_lists = self.read_images(normal_dir)

        if self.load_depth:
            self.depth_lists = self.read_depth(
                os.path.join(self.root, "sapiens/depths")
            )

        if smpl_type == "smpl":
            self.smpl_params = self.load_smpl_param(opt, self.root, split)

        elif smpl_type == "smplx":
            self.smpl_params = self.load_smplx_param(opt, self.root, split)

        self.bg_lists = []

        for idx in range(len(self.img_lists)):
            if self.random_bg:
                bgcolor = (np.random.rand(3)).astype(np.float32)
            else:
                bgcolor = np.array(self.bgcolor, dtype=np.float32)

            self.bg_lists.append(bgcolor)
            img = self.img_lists[idx]
            msk = self.msk_lists[idx]
            normal = self.normal_lists[idx]

            self.img_lists[idx] = (
                img * msk[..., None] + (1 - msk[..., None]) * bgcolor[None, None, :]
            )
            self.normal_lists[idx] = (
                normal * msk[..., None] + (1 - msk[..., None]) * bgcolor[None, None, :]
            )

            if self.load_depth:
                depth = self.depth_lists[idx]  # (H, W)
                m = msk > 0
                depth[m] = (depth[m] - depth[m].min()) / (
                    depth[m].max() - depth[m].min()
                )
                if self.reverse_depth:
                    depth[m] = 1.0 - depth[m]
                depth[~m] = 0.0
                self.depth_lists[idx] = depth

    def load_camera(self, opt):
        logging.info("loading camera params...")

        if self.smpl_type == "smpl":
            camera = np.load(os.path.join(self.root, "cameras.npz"))
            extrinsic = camera["extrinsic"].astype(np.float32)
            intrinsic = camera["intrinsic"].astype(np.float32)

            camera_params = get_camera_params(
                intrinsic=intrinsic / self.downscale,
                extrinsic=torch.tensor(extrinsic),
                height=int(camera["height"] / self.downscale),
                width=int(camera["width"] / self.downscale),
            )
        elif self.smpl_type == "smplx":
            width = opt.width
            height = opt.height
            camera_params = get_smplx_camera_params(width=width, height=height)

        return camera_params

    def load_smpl_param(self, opt, root, split):
        logging.info("loading smpl params...")
        use_refined_pose = opt.get("use_refined_pose", False)
        instant_avatar_skip = opt.get("intant_avatar_skip", 4)

        if use_refined_pose:
            smpl_data = dict(
                np.load(os.path.join(root, "poses", f"anim_nerf_{split}.npz"))
            )
            # smpl_data = dict(np.load(os.path.join(root, "poses", f"anim_nerf_train.npz")))
            # smpl_data_test = dict(np.load(os.path.join(root, "poses", f"anim_nerf_test.npz")))
            smpl_params = {}
            for k, v in smpl_data.items():
                smpl_params[k] = torch.tensor(v, dtype=torch.float32)
                # v1 = torch.tensor(smpl_data[k], dtype=torch.float32)
                # v2 = torch.tensor(smpl_data_test[k], dtype=torch.float32)
                # v = torch.cat([v1, v2], dim=0)
                # smpl_params[k] = v

                if k != "betas":
                    smpl_params[k] = smpl_params[k][
                        0 : v.shape[0] : self.skip // instant_avatar_skip
                    ]
            smpl_params["full_pose"] = torch.cat(
                [
                    smpl_params["global_orient"],
                    smpl_params["body_pose"],
                ],
                axis=1,
            )

        else:
            use_optimized_pose = opt.get("use_optimized_pose", True)
            if use_optimized_pose:
                logging.info("load poses_optimized.npz")
                pose_path = os.path.join(root, "poses_optimized.npz")
            else:
                logging.info("load poses.npz")
                pose_path = os.path.join(root, "poses.npz")

            smpl_data = dict(np.load(pose_path))
            B = smpl_data["transl"].shape[0]

            smpl_params = {
                "betas": torch.tensor(smpl_data["betas"], dtype=torch.float32).reshape(
                    1, 10
                ),
                "transl": torch.tensor(smpl_data["transl"], dtype=torch.float32),
            }
            if "thetas" in smpl_data:
                smpl_params["full_pose"] = torch.tensor(
                    smpl_data["thetas"], dtype=torch.float32
                )
            else:
                smpl_params["full_pose"] = torch.cat(
                    [
                        torch.tensor(smpl_data["global_orient"], dtype=torch.float32),
                        torch.tensor(smpl_data["body_pose"], dtype=torch.float32),
                    ],
                    axis=1,
                )
            for k, v in smpl_params.items():
                if k != "betas":
                    smpl_params[k] = v[self.start : self.end : self.skip]

        return smpl_params

    def load_smplx_param(self, opt, root, split):
        """
        pymafx_data.files: ['betas', 'body_pose', 'global_orient', 'smpl_verts', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'exp']
        """
        logging.info(f"loading smplx params...")
        path = os.path.join(root, "pymafx_result.npz")
        pymafx_data = np.load(path)

        B = pymafx_data["body_pose"].shape[0]

        eyp_pose = torch.eye(3).reshape(1, 1, 3, 3).repeat(B, 2, 1, 1).numpy()

        full_pose = np.concatenate(
            [
                pymafx_data["global_orient"],
                pymafx_data["body_pose"],
                pymafx_data["jaw_pose"],
                eyp_pose,
                pymafx_data["left_hand_pose"],
                pymafx_data["right_hand_pose"],
            ],
            axis=1,
        )

        smplx_params = {
            "betas": pymafx_data["betas"],  # (N, 10)
            "full_pose": full_pose,  # (N, 54, 3)
            "transl": pymafx_data["transl"],
            "scale": pymafx_data["scale"],
        }

        for k, v in smplx_params.items():
            v = v[self.start : self.end : self.skip]
            smplx_params[k] = torch.tensor(v, dtype=torch.float32)

        return smplx_params

    def read_images(self, img_dir):
        logging.info(f"loading image ...")
        img_name_lists = sorted(glob.glob(f"{img_dir}/*.png"))
        img_name_lists = img_name_lists[self.start : self.end : self.skip]
        img_lists = []

        for img_path in tqdm(img_name_lists, desc="loading image"):
            img_pil = Image.open(img_path).convert("RGB")
            img = np.array(img_pil.resize((self.width, self.height))) / 255.0
            img_lists.append(img)

        return img_lists

    def read_depth(self, dep_dir):
        logging.info(f"loading depth...")
        dep_name_lists = sorted(glob.glob(f"{dep_dir}/*.png"))
        dep_name_lists = dep_name_lists[self.start : self.end : self.skip]
        dep_lists = []

        for dep_path in tqdm(dep_name_lists, desc="loading depth"):
            dep_pil = Image.open(dep_path).convert("L")
            dep = np.array(dep_pil.resize((self.width, self.height)))
            dep = dep / 255.0
            dep_lists.append(dep)

        return dep_lists

    def read_masks(self, msk_dir):
        msk_name_lists = sorted(glob.glob(f"{msk_dir}/*"))
        msk_ext = msk_name_lists[0].split(".")[-1]
        logging.info(f"loading mask...")
        msk_name_lists = msk_name_lists[self.start : self.end : self.skip]
        msk_lists = []

        for msk_path in tqdm(msk_name_lists, desc="loading mask"):
            if msk_ext == "png":
                msk_pil = Image.open(msk_path).convert("L")
            elif msk_ext == "npy":
                msk_pil = Image.fromarray(np.load(msk_path) * 255).convert("L")

            msk = np.array(msk_pil.resize((self.width, self.height)))
            msk = msk / 255.0
            msk_lists.append(msk)

        return msk_lists

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, idx):

        smpl_params = {}
        for k, v in self.smpl_params.items():
            if k == "betas":
                smpl_params[k] = v[0]
            else:
                smpl_params[k] = v[idx]

        ret = {
            "idx": idx,
            "img": torch.tensor(self.img_lists[idx], dtype=torch.float32).permute(
                2, 0, 1
            ),  # (3, H, W)
            "mask": torch.tensor(
                self.msk_lists[idx][None], dtype=torch.float32
            ),  # (1, H, W)
            "normal": torch.tensor(self.normal_lists[idx], dtype=torch.float32).permute(
                2, 0, 1
            ),  # (3, H, W)
            "bgcolor": torch.tensor(self.bg_lists[idx]),
            "smpl_params": smpl_params,
            "camera_params": self.camera_params,
        }

        if self.load_depth:
            ret["depth"] = torch.tensor(
                self.depth_lists[idx][None], dtype=torch.float32
            )

        return ret

    def get_init_beta(self):
        return self.smpl_params["betas"][0:1]

    def get_smpl_pose(self):
        pose = self.smpl_params["body_pose"]
        return pose


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("test HumanDataset")
    opt = OmegaConf.create(
        {
            "split": {
                "train": {"start": 0, "end": 100, "skip": 10},
                "test": {"start": 100, "end": 200, "skip": 10},
            },
            "msk_ext": "npy",
        }
    )
    dataroot = "./data/peoplesnapshot"
    train_dataset = HumanDataset(data_root=dataroot, split="train", opt=opt)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    for i, batch_data in enumerate(dataloader):
        print(batch_data)
        break
    # print(ret)
    test_dataset = HumanDataset(data_root=dataroot, split="test", opt=opt)
    ret = test_dataset[0]
