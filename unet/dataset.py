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
        self.images = sorted(os.listdir(os.path.join(dir, "images_" + split)))
        self.annotation = sorted(os.listdir(os.path.join(dir, "annotation_" + split)))
        self.split = split
        self.dir = dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        image = cv2.imread(os.path.join(self.dir, "images_" + self.split, self.images[ix]))

        mask = cv2.imread(os.path.join(self.dir, "annotation_" + self.split, self.annotation[ix]), cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask > 127, 255, 0)
        
        image = image.astype(np.float32) / 255
        mask = mask.astype(np.float32) / 255

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, mask

def get_dataloaders():
    dir = "data"
    
    # Define Albumentations transformations
    transform = A.Compose([
        A.OneOf([
            A.RandomRain(p=0.5),
            A.RandomSnow(p=0.5),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Add normalization

        ToTensorV2(),
        # You can add more Albumentations transformations here
    ])
    
    trn_ds = SegmentationData(dir, 'train', transform=transform)
    val_ds = SegmentationData(dir, 'test', transform=transform)

    trn_dl = DataLoader(trn_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=True)

    return trn_dl, val_dl
