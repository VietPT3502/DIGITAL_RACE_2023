import torch
import torch.nn as nn


def UnetLoss(preds, targets):
    smooth = 1.0  # Smoothing factor to avoid division by zero
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = targets.contiguous().view(targets.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1) + smooth
    den = torch.sum(predict.pow(2) + target.pow(2), dim=1) + smooth

    loss = 1 - num / den
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    return loss.mean(), acc



