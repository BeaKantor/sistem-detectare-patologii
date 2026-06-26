

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from preprocess import (
    get_train_transforms,
    get_strong_train_transforms,
    get_val_transforms,
    IMBALANCED_DISEASES,
)

DATA_DIR   = r"C:\xray-project\data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")

DISEASES = [
    'enlarged cardiomediastinum', 'cardiomegaly', 'atelectasis',
    'consolidation', 'lung edema', 'fracture', 'lung lesion',
    'pleural effusion', 'pneumonia', 'pneumothorax',
    'support device', 'lung opacity', 'pleural other'
]

NUM_CLASSES = len(DISEASES)

OVERSAMPLE_FACTOR = 3.0


class ChestXrayDataset(Dataset):


    def __init__(self, data_dir, images_dir, split, is_training=False):
        self.images_dir  = images_dir
        self.is_training = is_training

        self.labels_df      = pd.read_csv(os.path.join(data_dir, f"{split}_labels_clean.csv"))
        self.probability_df = pd.read_csv(os.path.join(data_dir, f"{split}_probability_clean.csv"))
        self.severity_df    = pd.read_csv(os.path.join(data_dir, f"{split}_severity_clean.csv"))
        self.location_df    = pd.read_csv(os.path.join(data_dir, f"{split}_location_clean.csv"))

        self.filenames = self.labels_df['subjectid_studyid'].apply(
            lambda p: os.path.basename(p)
        ).tolist()

        self.labels_arr      = self.labels_df[DISEASES].values.astype(np.float32)
        self.probability_arr = self.probability_df[DISEASES].values.astype(np.float32)
        self.severity_arr    = self.severity_df[DISEASES].values.astype(np.float32)
        self.location_arr    = self.location_df[DISEASES].fillna('nan').values.astype(str)

        self.use_strong_aug  = self._compute_strong_aug_flags()
        self.train_tfm       = get_train_transforms()
        self.strong_train_tfm = get_strong_train_transforms()
        self.val_tfm         = get_val_transforms()

    def _compute_strong_aug_flags(self) -> np.ndarray:
        flags = np.zeros(len(self.labels_df), dtype=bool)
        disease_indices = {
            d: DISEASES.index(d)
            for d in IMBALANCED_DISEASES
            if d in DISEASES
        }
        for img_idx in range(len(self.labels_df)):
            for disease, col_idx in disease_indices.items():
                if self.labels_arr[img_idx, col_idx] == 1.0:
                    flags[img_idx] = True
                    break
        return flags

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
 
        img_path = os.path.join(self.images_dir, self.filenames[idx])
        img_bgr  = cv2.imread(img_path)

        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.is_training:
            if self.use_strong_aug[idx]:
                image = self.strong_train_tfm(image=img_rgb)["image"]
            else:
                image = self.train_tfm(image=img_rgb)["image"]
        else:
            image = self.val_tfm(image=img_rgb)["image"]

        labels      = torch.tensor(self.labels_arr[idx],      dtype=torch.float32)
        probability = torch.tensor(self.probability_arr[idx], dtype=torch.float32)
        severity    = torch.tensor(self.severity_arr[idx],    dtype=torch.float32)
        location    = self.location_arr[idx].tolist()

        return image, labels, probability, severity, location


def build_sampler(dataset: ChestXrayDataset) -> WeightedRandomSampler:

    weights = np.where(dataset.use_strong_aug, OVERSAMPLE_FACTOR, 1.0)
    sampler = WeightedRandomSampler(
        weights     = torch.from_numpy(weights).double(),
        num_samples = len(dataset),
        replacement = True,
    )
    return sampler


def get_dataloaders(
    data_dir:    str = DATA_DIR,
    images_dir:  str = IMAGES_DIR,
    batch_size:  int = 32,
    num_workers: int = 0,
) -> tuple:

    train_dataset = ChestXrayDataset(data_dir, images_dir, "train", is_training=True)
    val_dataset   = ChestXrayDataset(data_dir, images_dir, "val",   is_training=False)
    test_dataset  = ChestXrayDataset(data_dir, images_dir, "test",  is_training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        sampler     = build_sampler(train_dataset),
        num_workers = num_workers,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    return train_loader, val_loader, test_loader