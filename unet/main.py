from engine import engine
from loss import UnetLoss
from config import config
from tqdm import tqdm
from model import make_model
from dataset import get_dataloaders
import cv2
import numpy as np
import torch
def run():
  trn_dl, val_dl = get_dataloaders()
  model, criterion, optimizer, scheduler = make_model()
  best_loss = 100
  for epoch in range(config.N_EPOCHS):
    print("####################")
    print(f"       Epoch: {epoch}   ")
    print("####################")
    train_losses = []
    val_losses = []
    current_lr = optimizer.param_groups[0]['lr']
    with tqdm(total=len(trn_dl)) as pbar:
        for bx, data in enumerate(trn_dl):
            # image, mask = data
            # image = image[0].detach().cpu().numpy()
            # image *= 255
            # image = image.astype(np.uint8)
            # image = np.transpose(image, (1, 2, 0)) 
            # print(image.shape)
            # print(type(image))
            # cv2.imshow("image", image)
            # cv2.waitKey(0)

            # mask = mask[0].detach().cpu().numpy()
            # mask *= 255
            # mask = mask.astype(np.uint8)
            # print(mask.shape)
            # print(type(mask))
            # cv2.imshow("mask", mask)
            # cv2.waitKey(0)
            # exit()
            train_loss, train_acc = engine.train_batch(model, data, optimizer, criterion)
            train_losses.append(train_loss)
            
            pbar.update(1)
    scheduler.step()
    mean_train_loss = sum(train_losses) / len(train_losses)
    print("train loss at epoch{}: {}, acc: {}, lr: {}".format(epoch + 1, mean_train_loss, train_acc, current_lr))
    with tqdm(total=len(val_dl)) as pbar:
        for bx, data in tqdm(enumerate(val_dl), total = len(val_dl)):
            val_loss, val_acc = engine.validate_batch(model, data, criterion)
            pbar.update(1)
            val_losses.append(val_loss)
    mean_val_loss = sum(val_losses) / len(val_losses)
    print("val loss at epoch{}: {}, acc: {}".format(epoch + 1, mean_val_loss, val_acc))

    if mean_val_loss < best_loss:
        best_loss = mean_val_loss
        torch.save(model.state_dict(), 'best_loss.pth')
        print("save best loss with loss {}".format(mean_val_loss))

    print()

if __name__ == "__main__":
    run()