

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMBALANCED_DISEASES = {
    'lung edema',                  # 1.3%
    'pneumonia',                   # 1.8%
    'fracture',                    # 2.0%
    'cardiomegaly',                # 3.2%
    'pneumothorax',                # 3.4%
    'enlarged cardiomediastinum',  # 3.5%
    'pleural other',               # 4.2%
    'pleural effusion',            # 9.0%
    'consolidation',               # 9.1%
}


def get_train_transforms() -> A.Compose:
    
    return A.Compose([
        A.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE),
            min_max_height=(int(IMAGE_SIZE * 0.85), IMAGE_SIZE),
            ratio=(0.9, 1.1),
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.4,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.4,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_strong_train_transforms() -> A.Compose:
    
    return A.Compose([
        A.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE),
            min_max_height=(int(IMAGE_SIZE * 0.75), IMAGE_SIZE),
            ratio=(0.85, 1.15),
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.6,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=0.08,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.6,
        ),
        A.GaussianBlur(
            blur_limit=(3, 7),
            p=0.3,
        ),
        A.GaussNoise(p=0.3),
        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=0.3,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])