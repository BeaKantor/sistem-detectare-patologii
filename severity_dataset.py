

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from preprocess import get_train_transforms, get_val_transforms

DATA_DIR   = r"C:\xray-project\data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")

DISEASES = [
    'enlarged cardiomediastinum', 'cardiomegaly', 'atelectasis',
    'consolidation', 'lung edema', 'fracture', 'lung lesion',
    'pleural effusion', 'pneumonia', 'pneumothorax',
    'support device', 'lung opacity', 'pleural other'
]


SEVERITY_COUNTS = {
    0: 1007396,
    1: 7652,
    2: 5071,
    3: 459,
}


class SeverityDataset(Dataset):
    

    def __init__(self, data_dir, images_dir, split, is_training=False):
        self.images_dir  = images_dir
        self.is_training = is_training
        self.transform   = get_train_transforms() if is_training else get_val_transforms()

        
        labels_df   = pd.read_csv(os.path.join(data_dir, f"{split}_labels_clean.csv"))
        severity_df = pd.read_csv(os.path.join(data_dir, f"{split}_severity_clean.csv"))

        
        filenames = labels_df['subjectid_studyid'].apply(
            lambda p: os.path.basename(p)
        ).tolist()

        labels_arr   = labels_df[DISEASES].values.astype(np.float32)
        severity_arr = severity_df[DISEASES].values.astype(np.float32)

        self.samples = []

        for img_idx in range(len(labels_df)):
            for disease_idx, disease in enumerate(DISEASES):
                label    = labels_arr[img_idx, disease_idx]
                severity = severity_arr[img_idx, disease_idx]

                if label != 1.0:
                    continue

                if np.isnan(severity):
                    continue

                self.samples.append((
                    filenames[img_idx],
                    disease_idx,
                    int(severity),
                ))

        print(f"  {split} severity dataset: {len(self.samples):,} samples "
              f"from confirmed present cases")

        sev_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for _, _, sev in self.samples:
            sev_counts[sev] += 1
        print(f"  Distribution: N/A={sev_counts[0]:,} "
              f"mild={sev_counts[1]:,} "
              f"moderate={sev_counts[2]:,} "
              f"severe={sev_counts[3]:,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
      
        filename, disease_idx, severity = self.samples[idx]

        img_path = os.path.join(self.images_dir, filename)
        img_bgr  = cv2.imread(img_path)

        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image   = self.transform(image=img_rgb)["image"]

        return (
            image,
            torch.tensor(disease_idx, dtype=torch.long),
            torch.tensor(severity,    dtype=torch.long),
        )


def build_severity_sampler(dataset: SeverityDataset) -> WeightedRandomSampler:
   
    total = sum(SEVERITY_COUNTS.values())
    class_weights = {
        k: total / (4 * v)
        for k, v in SEVERITY_COUNTS.items()
    }

    sample_weights = torch.tensor(
        [class_weights[sev] for _, _, sev in dataset.samples],
        dtype=torch.double,
    )

    return WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(dataset),
        replacement = True,
    )


def get_severity_dataloaders(
    data_dir:    str = DATA_DIR,
    images_dir:  str = IMAGES_DIR,
    batch_size:  int = 32,
    num_workers: int = 0,
) -> tuple:
   
    train_ds = SeverityDataset(data_dir, images_dir, "train", is_training=True)
    val_ds   = SeverityDataset(data_dir, images_dir, "val",   is_training=False)
    test_ds  = SeverityDataset(data_dir, images_dir, "test",  is_training=False)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        sampler     = build_severity_sampler(train_ds),
        num_workers = num_workers,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    return train_loader, val_loader, test_loader