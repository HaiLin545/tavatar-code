import os
import torch
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

from utils.cam_utils import get_camera_params, get_smplx_camera_params
import logging
from tqdm import tqdm


class NeumanDataset(Dataset):
    def __init__(
        self,
        root="D:\\SMIL\\datasets\\custom_dataset\\dataset\\neuman",
        subject="bike",
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
        self.height = self.img_height
        self.width = self.img_width

        self.random_bg = split == "train" and opt.get("random_bg", False)
        self.bgcolor = opt.get("bgcolor", [1.0, 1.0, 1.0])
        self.img_lists = self.read_images(os.path.join(self.root, "images"))
        mask_dir = os.path.join(self.root, "masks")
        normal_dir = os.path.join(self.root, "normals")
        self.msk_lists = self.read_masks(mask_dir)
        self.normal_lists = self.read_images(normal_dir)

        # 让img、mask、normal从左向右拼接然后导出一个视频，我要查看是否读取正确
        # self.export_verification_video()

        if smpl_type == "smpl":
            self.smpl_params = self.load_smpl_param(opt, self.root, split)

        elif smpl_type == "smplx":
            self.smpl_params = self.load_smplx_param(opt, self.root, split)

        # stress test, add noise to body_pose
        self.noise = opt.get("noise", 0.0)
        if self.noise > 0.0:
            noise = torch.randn_like(self.smpl_params["full_pose"][:, 3:]) * self.noise
            self.smpl_params["full_pose"][:, 3:] += noise

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

    def load_camera(self, opt):
        logging.info("loading camera params from COLMAP txt files...")

        sparse_dir = os.path.join(self.root, "sparse")
        cameras_txt = os.path.join(sparse_dir, "cameras.txt")
        images_txt = os.path.join(sparse_dir, "images.txt")

        # Parse cameras.txt for intrinsics (PINHOLE model)
        with open(cameras_txt, "r") as f:
            lines = f.readlines()

        # Find the camera line (skip comments)
        for line in lines:
            if line.startswith("#") or line.strip() == "":
                continue
            # Format: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
            # For PINHOLE: camera_id PINHOLE width height fx fy cx cy
            parts = line.strip().split()
            if parts[1] == "PINHOLE":
                width = int(parts[2])
                height = int(parts[3])
                fx = float(parts[4])
                fy = float(parts[5])
                cx = float(parts[6])
                cy = float(parts[7])
                break

        # Build intrinsic matrix
        self.cam_intrinsic = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
        )

        # Parse images.txt for extrinsics
        # Format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        with open(images_txt, "r") as f:
            lines = f.readlines()

        # Collect all image lines (skip comments and point lines)
        image_data = []
        for line in lines:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split()
            # Image lines start with numeric ID and have at least 10 parts
            if len(parts) >= 10:
                try:
                    image_id = int(parts[0])
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    image_name = parts[9]
                    image_data.append(
                        {
                            "id": image_id,
                            "quat": np.array([qw, qx, qy, qz]),
                            "trans": np.array([tx, ty, tz]),
                            "name": image_name,
                        }
                    )
                except ValueError:
                    continue

        # Sort by image name to ensure correct order
        image_data.sort(key=lambda x: x["name"])

        # Apply start/end/skip indexing
        image_data = image_data[self.start : self.end : self.skip]

        # Convert quaternion + translation to extrinsic matrices
        extrinsics = []

        # COLMAP to OpenGL coordinate transform
        # COLMAP: Y down, Z forward (OpenCV convention)
        # OpenGL: Y up, Z backward
        colmap_to_opengl = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32
        )

        for img in image_data:
            # Quaternion to rotation matrix
            qw, qx, qy, qz = img["quat"]
            R = self._quat_to_rotation_matrix(qw, qx, qy, qz)
            t = img["trans"]

            # Build world-to-camera matrix (COLMAP convention)
            w2c_colmap = np.eye(4, dtype=np.float32)
            w2c_colmap[:3, :3] = R
            w2c_colmap[:3, 3] = t

            # Convert COLMAP w2c to OpenGL w2c
            # First convert to c2w in COLMAP space, then apply coordinate transform
            c2w_colmap = np.linalg.inv(w2c_colmap)
            c2w_opengl = c2w_colmap @ colmap_to_opengl
            w2c_opengl = np.linalg.inv(c2w_opengl)

            extrinsics.append(w2c_opengl)

        self.img_height = height
        self.img_width = width
        if self.downscale > 1:
            self.cam_intrinsic = self.cam_intrinsic / self.downscale
            self.img_height = int(self.img_height / self.downscale)
            self.img_width = int(self.img_width / self.downscale)

        camera_params = []
        for extrinsic in extrinsics:
            cam_param = get_camera_params(
                intrinsic=self.cam_intrinsic,
                extrinsic=torch.tensor(extrinsic),
                height=int(self.img_height),
                width=int(self.img_width),
            )
            camera_params.append(cam_param)
        return camera_params

    def _quat_to_rotation_matrix(self, w, x, y, z):
        """Convert quaternion to 3x3 rotation matrix"""
        R = np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * w * z,
                    2 * x * z + 2 * w * y,
                ],
                [
                    2 * x * y + 2 * w * z,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * w * x,
                ],
                [
                    2 * x * z - 2 * w * y,
                    2 * y * z + 2 * w * x,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ],
            dtype=np.float32,
        )
        return R

    def load_smpl_param(self, opt, root, split):
        logging.info("loading smpl params...")
        smpl_params_path = os.path.join(root, "smpl_optimized_aligned_scale.npz")
        smpl_data = np.load(smpl_params_path)
        smpl_data = {f: smpl_data[f] for f in smpl_data.files}

        smpl_params = []
        for k, v in smpl_data.items():
            smpl_param = {}
            smpl_param[k] = torch.tensor(
                v[self.start : self.end : self.skip], dtype=torch.float32
            )
            smpl_params.append(smpl_param)

        for smpl_param in smpl_params:
            smpl_param["full_pose"] = torch.cat(
                [smpl_param["global_orient"], smpl_param["body_pose"]], dim=1
            )

        # smpl_params_path = f'{dataset_path}/4d_humans/'
        # smpl_params = np.load(smpl_params_path)
        # load smpl params from npz files
        # pose_paths = sorted(glob.glob(f"{pose_dir}/*.npz"))[
        #     self.start : self.end : self.skip
        # ]
        # smpl_params = []
        # for pose_path in tqdm(pose_paths, desc="loading smpl params"):
        #     smpl_data = dict(np.load(pose_path, allow_pickle=True))
        #     smpl_data = smpl_data["results"][0]
        #     smpl_param = {
        #         "betas": torch.tensor(smpl_data["betas"], dtype=torch.float32),
        #         "transl": torch.tensor(smpl_data["trans"], dtype=torch.float32),
        #         "full_pose": torch.tensor(
        #             smpl_data["poses"],
        #             dtype=torch.float32,
        #         ),
        #     }
        #     smpl_params.append(smpl_param)

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
        img = torch.tensor(self.img_lists[idx], dtype=torch.float32).permute(
            2, 0, 1
        )  # (3, H, W)
        msk = torch.tensor(self.msk_lists[idx][None], dtype=torch.float32)  # (1, H, W)
        normal = torch.tensor(self.normal_lists[idx], dtype=torch.float32).permute(
            2, 0, 1
        )  # (3, H, W)
        bgcolor = torch.tensor(self.bg_lists[idx])

        ret = {
            "idx": idx,
            "img": img,
            "mask": msk,
            "normal": normal,
            "bgcolor": bgcolor,
            "smpl_params": self.smpl_params[idx],
            "camera_params": self.camera_params[idx],
        }

        return ret

    def get_init_beta(self):
        return self.smpl_params["betas"][0:1]

    def get_smpl_pose(self):
        pose = self.smpl_params["body_pose"]
        return pose

    def export_verification_video(self):
        """导出img、mask、normal水平拼接的视频用于验证数据读取"""
        logging.info("Exporting verification video...")

        # 创建输出目录
        output_dir = os.path.join("verification_videos")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"{self.label}_{self.split}_verification.mp4"
        )

        # 获取视频参数
        height, width = self.height, self.width
        fps = 10

        # 创建VideoWriter (3倍宽度用于拼接)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width * 3, height))

        for idx in tqdm(range(len(self.img_lists)), desc="Creating verification video"):
            img = self.img_lists[idx]
            msk = self.msk_lists[idx]
            normal = self.normal_lists[idx]

            # 将mask转换为3通道
            msk_3ch = np.stack([msk, msk, msk], axis=-1)

            # 水平拼接 img, mask, normal
            concatenated = np.concatenate([img, msk_3ch, normal], axis=1)

            # 转换为BGR格式 (OpenCV使用BGR)
            concatenated_bgr = (concatenated * 255).astype(np.uint8)
            concatenated_bgr = cv2.cvtColor(concatenated_bgr, cv2.COLOR_RGB2BGR)

            video_writer.write(concatenated_bgr)

        video_writer.release()
        logging.info(f"Verification video saved to: {output_path}")


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
    dataroot = "/mnt/cephfs/home/luohailin/code/4d_human/GoM2/data/peoplesnapshot"
    train_dataset = HumanDataset(data_root=dataroot, split="train", opt=opt)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    for i, batch_data in enumerate(dataloader):
        print(batch_data)
        break
    # print(ret)
    test_dataset = HumanDataset(data_root=dataroot, split="test", opt=opt)
    ret = test_dataset[0]
