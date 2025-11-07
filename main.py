import os
import toml
import argparse
import torch
import time
from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything, Trainer
from utils.config import merge_args_to_config, DotDict, merge_dicts
from tavatar import TavatarModel
from dataset import get_dataset
from dataset.AnimateDataset import AnimateDataset
from utils.smpl import gen_canonical_pose
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import Timer, ModelCheckpoint
from utils.output import save_predict, save_ply

timer = Timer()


def train(cfg, model: TavatarModel, trainer: Trainer):
    train_dataset = get_dataset(cfg, split="train")
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=4)
    trainer.fit(model=model, train_dataloaders=train_dataloader)

    print(f"fit done, fit time = {timer.time_elapsed('train'):.2f} seconds")


def predict(cfg, model: TavatarModel, trainer: Trainer):

    canonical_pose = gen_canonical_pose(
        smpl_type=cfg.smpl.type, canonical_type=cfg.smpl.canonical_type
    )
    canonical_dataset = AnimateDataset(
        width=512, height=512, pose=canonical_pose.repeat(120, 1), is360=True
    )
    canonical_dataloader = DataLoader(canonical_dataset, shuffle=False, num_workers=4)
    outputs = trainer.predict(model=model, dataloaders=canonical_dataloader)

    save_dir = os.path.join(trainer.logger.log_dir, f"predict_{trainer.current_epoch}")
    os.makedirs(save_dir, exist_ok=True)
    print(
        f"predict canonical done, predict time = {timer.time_elapsed('predict'):.2f} seconds"
    )
    save_predict(outputs, save_dir, "canonical")

    poses_path = [
        "./novel_poses/cxk.npy",
        # "./novel_poses/aist_demo.npy",
        # "./novel_poses/walking.npy",
    ]
    for i, pose_path in enumerate(poses_path):
        animate_dataset = AnimateDataset(
            width=512, height=512, pose_path=pose_path, is360=False
        )
        animate_dataloader = DataLoader(animate_dataset, shuffle=False)
        outputs = trainer.predict(model=model, dataloaders=animate_dataloader)
        save_predict(outputs, save_dir, os.path.basename(pose_path).split(".")[0])
        print(
            f"predict done, pose = {pose_path}, predict time = {timer.time_elapsed('predict'):.2f} seconds"
        )


def predict_pose(cfg, model: TavatarModel, trainer: Trainer, pose_path: str):

    animate_dataset = AnimateDataset(
        width=512, height=512, pose_path=pose_path, is360=True
    )
    animate_dataloader = DataLoader(animate_dataset, shuffle=False)

    start_time = time.monotonic()
    outputs = trainer.predict(model=model, dataloaders=animate_dataloader)
    end_time = time.monotonic()
    total_time = end_time - start_time  #

    save_dir = os.path.join(trainer.logger.log_dir, f"predict_{trainer.current_epoch}")
    os.makedirs(save_dir, exist_ok=True)
    save_predict(outputs, save_dir, os.path.basename(pose_path).split(".")[0])

    fps = len(animate_dataset) / total_time
    print(
        f"predict done, pose = {pose_path}, predict time = {total_time:.2f} seconds, fps={fps:.1f} ",
    )


def test(cfg, model: TavatarModel, trainer: Trainer):
    test_datasets = get_dataset(cfg, split="test")
    test_dataloaders = [
        DataLoader(test_dataset, shuffle=False, num_workers=4)
        for test_dataset in test_datasets
    ]
    trainer.test(model=model, dataloaders=test_dataloaders)


def main(args, cfg):
    seed_everything(cfg["seed"], workers=True)

    logger = TensorBoardLogger(
        save_dir=cfg.log_dir, version=cfg.dataset.subject, name=""
    )
    ckpt_callback = ModelCheckpoint(save_last=True)
    trainer = Trainer(
        logger=logger,
        callbacks=[timer, ckpt_callback],
        detect_anomaly=True,
        **cfg.trainer,
    )

    if args.ckpt is not None:
        print(f"load from {args.ckpt}")
        tavatarModel = TavatarModel.load_from_checkpoint(args.ckpt, cfg=cfg)
    else:
        tavatarModel = TavatarModel(cfg)

    if args.train:
        print("---------- Start Training ----------")
        train(cfg, tavatarModel, trainer)
        print("---------- End Training ----------")

    if args.test:
        print("---------- Start Testing ----------")
        test(cfg, tavatarModel, trainer)
        print("---------- End Testing ----------")

    if args.animate:
        if args.pose_path == "all":
            poses_path = [
                "./novel_poses/aist_demo.npy",
                "./novel_poses/poses/da_pose_smpl.npy",
                "./novel_poses/poses/t_pose_smpl.npy",
            ]
            for pose_path in poses_path:
                predict_pose(cfg, tavatarModel, trainer, pose_path)
        else:
            predict_pose(cfg, tavatarModel, trainer, args.pose_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./config/xhuman_16.toml")
    parser.add_argument("--device", type=int, default=5)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--resume_dir", type=str, default=None)
    # parser.add_argument(
    #     "--ckpt",
    #     type=str,
    #     default="./output/people_snapshot/male-3-casual/checkpoints/epoch=1-step=46.ckpt",
    # )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--pose_path", type=str, default="all")

    args, unknown = parser.parse_known_args()

    if args.resume_dir is not None:
        args.ckpt = os.path.join(args.resume_dir, "checkpoints", "last.ckpt")
        args.cfg = os.path.join(args.resume_dir, "config.toml")

    default_cfg = toml.load("./config/default.toml")
    user_cfg = toml.load(args.cfg)
    cfg = merge_dicts(default_cfg, user_cfg)
    cfg = merge_args_to_config(cfg, unknown)
    cfg = DotDict(cfg)

    if args.device is not None:
        cfg.trainer["devices"] = [args.device]

    log_dir = os.path.join(cfg.output, cfg.exp_name, cfg.dataset.name)
    os.makedirs(log_dir, exist_ok=True)
    cfg.log_dir = log_dir

    main(args, cfg)

    print("all done")
