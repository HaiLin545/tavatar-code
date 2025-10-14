import os
import toml
import argparse
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything, Trainer
from utils.config import merge_args_to_config, DotDict, merge_dicts
from tavatar import TavatarModel
from dataset import get_dataset
from dataset.AnimateDataset import AnimateDataset
from utils.smpl import gen_canonical_pose
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import Timer
from utils.output import save_predict, save_ply

timer = Timer()


def train(cfg, model: TavatarModel, trainer: Trainer):
    train_dataset = get_dataset(cfg, split="train")
    train_dataloader = DataLoader(train_dataset, shuffle=True)
    trainer.fit(model=model, train_dataloaders=train_dataloader)

    print(f"fit done, fit time = {timer.time_elapsed('train'):.2f} seconds")


def predict(cfg, model: TavatarModel, trainer: Trainer):

    canonical_pose = gen_canonical_pose(
        smpl_type=cfg.smpl.type, canonical_type=cfg.smpl.canonical_type
    )
    canonical_dataset = AnimateDataset(
        width=512, height=512, pose=canonical_pose.repeat(120, 1), is360=True
    )
    canonical_dataloader = DataLoader(canonical_dataset, shuffle=False)
    outputs = trainer.predict(model=model, dataloaders=canonical_dataloader)

    save_dir = os.path.join(trainer.logger.log_dir, "predict")
    print(
        f"predict canonical done, predict time = {timer.time_elapsed('predict'):.2f} seconds"
    )
    save_predict(outputs, save_dir, "canonical")


    poses_path = [
        "./novel_poses/aist_demo.npy",
        "./novel_poses/walking.npy",
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


def test(cfg, model: TavatarModel, trainer: Trainer):
    test_datasets = get_dataset(cfg, split="test")
    test_dataloaders = [
        DataLoader(test_dataset, shuffle=False) for test_dataset in test_datasets
    ]
    trainer.test(model=model, dataloaders=test_dataloaders)


def main(cfg):
    seed_everything(cfg["seed"], workers=True)

    logger = TensorBoardLogger(
        save_dir=cfg.log_dir, version=cfg.dataset.subject, name=""
    )
    trainer = Trainer(
        logger=logger, callbacks=[timer], detect_anomaly=True, **cfg.trainer
    )

    if cfg.ckpt is not None:
        print(f"load from {cfg.ckpt}")
        tavatarModel = TavatarModel.load_from_checkpoint(cfg.ckpt, cfg=cfg)
        predict(cfg, tavatarModel, trainer)
    else:
        tavatarModel = TavatarModel(cfg)
        train(cfg, tavatarModel, trainer)
        test(cfg, tavatarModel, trainer)
        predict(cfg, tavatarModel, trainer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./config/people_m3c.toml")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    # parser.add_argument(
    #     "--ckpt",
    #     type=str,
    #     default="./output/people_snapshot/male-3-casual/checkpoints/epoch=1-step=46.ckpt",
    # )

    args, unknown = parser.parse_known_args()

    default_cfg = toml.load("./config/default.toml")
    user_cfg = toml.load(args.cfg)
    cfg = merge_dicts(default_cfg, user_cfg)
    cfg = merge_args_to_config(cfg, unknown)
    cfg = DotDict(cfg)

    if args.device is not None:
        cfg.trainer["devices"] = [args.device]

    log_dir = os.path.join(cfg.output, cfg.dataset.name)
    os.makedirs(log_dir, exist_ok=True)
    cfg.log_dir = log_dir
    cfg.ckpt = args.ckpt

    main(cfg)
