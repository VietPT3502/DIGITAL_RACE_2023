import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .config import config

class SegmentationData(Dataset):
    def __init__(self, dir, split, transform=None):
        self.transform = transform
        self.images = sorted(os.listdir(os.path.join(dir, split + "_image")))
        self.annotation = sorted(os.listdir(os.path.join(dir, split + "_mask")))
        self.split = split
        self.dir = dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        image = cv2.imread(os.path.join(self.dir, self.split + "_image", self.images[ix]))

        mask = cv2.imread(os.path.join(self.dir, self.split + "_mask", self.annotation[ix]), cv2.IMREAD_GRAYSCALE)
        
        image = image.astype(np.float32) / 255
        mask = mask.astype(np.float32) / 255

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, mask

def get_dataloaders():
    dir = "data"
    
    # Define Albumentations transformations
    transform_train = A.Compose([
        A.RandomBrightnessContrast(p=0.3),
        # Apply image blurring
        A.OneOf([
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.3),
        A.OneOf([
            A.RandomRain(),
            A.RandomSnow(),
        ], p=0.3),

        ToTensorV2(),
        # You can add more Albumentations transformations here
    ])
    transform_val = A.Compose([

        ToTensorV2(),
        # You can add more Albumentations transformations here
    ])
    
    trn_ds = SegmentationData(dir, 'train', transform=transform_train)
    val_ds = SegmentationData(dir, 'test', transform=transform_val)

    trn_dl = DataLoader(trn_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=True)

    return trn_dl, val_dl
