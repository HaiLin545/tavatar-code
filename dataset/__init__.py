import logging
from dataset.HumanDataset import HumanDataset
from dataset.XhumanDataset import XhumanDataset_Multi
import os


def get_dataset(cfg, split="train"):

    if split == "train":
        return get_train_dataset(cfg)
    elif split == "test":
        return get_test_datasets(cfg)


def get_train_dataset(cfg):
    dataset_name = cfg.dataset.name
    logging.info(f"Load train dataset: {dataset_name}")
    dataset_root = os.path.join(cfg.dataset.dataset_dir, cfg.dataset.name)
    
    if dataset_name == "xhuman":
        data_split_list = cfg.dataset.get("train_list", [])
        train_dataset = XhumanDataset_Multi(
            data_root=dataset_root,
            subject=cfg.dataset.subject,
            split="train",
            data_split_list=data_split_list,
            smpl_type=cfg.model.smpl_type,
            cfg=cfg.dataset,
        )
        cfg.model.smpl_gender = train_dataset.datasets[0].gender

    else:  # dataset_name == "people_snapshot" or dataset_name == "custom":
        train_dataset = HumanDataset(
            root=dataset_root,
            subject=cfg.dataset.subject,
            split="train",
            smpl_type=cfg.smpl.type,
            opt=cfg.dataset,
        )

    return train_dataset


def get_test_datasets(cfg):
    dataset_name = cfg.dataset.name
    logging.info(f"Load test dataset: {dataset_name}")
    dataset_root = os.path.join(cfg.dataset.dataset_dir, cfg.dataset.name)
    
    test_datasets = []
    if dataset_name == "xhuman":
        data_split_list = cfg.dataset.get("test_list", [])
        test_dataset = XhumanDataset_Multi(
            data_root=dataset_root,
            subject=cfg.dataset.subject,
            split="test",
            data_split_list=data_split_list,
            smpl_type=cfg.model.smpl_type,
            cfg=cfg.dataset,
        )
        cfg.model.smpl_gender = test_dataset.datasets[0].gender
        test_datasets.append(test_dataset)
    else:  # dataset_name == "people_snapshot" or dataset_name == "custom":
        dataset = HumanDataset(
            root=dataset_root,
            subject=cfg.dataset.subject,
            split="test",
            smpl_type=cfg.smpl.type,
            opt=cfg.dataset,
        )
        test_datasets.append(dataset)

    logging.info(f"Load {len(test_datasets)} test dataset: {dataset_name}")

    return test_datasets
