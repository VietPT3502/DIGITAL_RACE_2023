import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import config
def dice_loss(preds, targets):
    smooth = 1.0  # Smoothing factor to avoid division by zero
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = targets.contiguous().view(targets.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1) + smooth
    den = torch.sum(predict.pow(2) + target.pow(2), dim=1) + smooth

    loss = 1 - num / den
    return loss.mean()

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def hinge(pred, label):
    signs = 2 * label - 1
    errors = 1 - pred * signs
    return errors


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore_index: label to ignore
    """
    logits = logits.contiguous().view(-1)
    labels = labels.contiguous().view(-1)
    mask = labels
    logits = logits[mask.bool()]
    labels = labels[mask.bool()]
    errors = hinge(logits, labels)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.elu(errors_sorted) + 1, grad)
    return loss

def binary_crossentropy_loss(preds, targets):
    return F.binary_cross_entropy_with_logits(preds, targets)


def SegmentationLoss(preds, targets):
    dice = dice_loss(preds, targets)
    lovasz = lovasz_hinge_flat(preds, targets)
    bce = binary_crossentropy_loss(preds, targets)
    return dice *config.DICE_COEF + bce + lovasz * config.LOVASZ_COEF


