from .engine import engine
from .loss import SegmentationLoss
from .config import config
from tqdm import tqdm
from .model import make_model
from .dataset import get_dataloaders
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
def save_table():
    trn_dl, val_dl = get_dataloaders()
    model, criterion, optimizer = make_model()
    model.load_state_dict(torch.load('best_loss.pth'))
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for bx, data in tqdm(enumerate(val_dl), total=len(val_dl)):
            im, mask = data
            im = im.to(config.DEVICE)
            mask = mask.to(config.DEVICE)
            _mask = model(im)[0]
            print(_mask.shape)

            # Save original image
            original_image = im[0].permute(1, 2, 0).detach().cpu().numpy()[:,:,0]
            cv2.imwrite("original_image.jpg", original_image * 255)

            # Save original mask
            original_mask = mask.permute(1, 2, 0).detach().cpu().numpy()[:,:,0]
            cv2.imwrite("original_mask.jpg", original_mask * 255)

            # Save predicted mask
            predicted_mask = _mask.permute(1, 2, 0).detach().cpu().numpy()[:,:,0]
            predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8) * 255
            cv2.imwrite("predicted_mask.jpg", predicted_mask_binary)

def predict(model, image):

    transform = A.Compose([


        ToTensorV2(),
        # You can add more Albumentations transformations here
    ])
    
    with torch.no_grad():
        image = image.astype(np.float32) / 255
        augmented = transform(image=image)
        image = augmented['image']
        image = image.unsqueeze(0).to(config.DEVICE)

        mask= model(image)
        predicted_mask = mask[0].permute(1, 2, 0).detach().cpu().numpy()[:,:,0]
        predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8) * 255
    return predicted_mask_binary


# model, criterion, optimizer, scheduler = make_model()
# model.load_state_dict(torch.load('best_loss.pth'))
# model.eval()  # Set the model to evaluation mode
# image = cv2.imread("/home/vietpt/vietpt/code/race/unet/data/test_image/new_00208.jpg")
# mask =predict(model, image)
# cv2.imshow("mask",mask)
# cv2.waitKey(0)