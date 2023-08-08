import torch
from .config import config

class engine():
  def train_batch(model, data, optimizer, criterion):
    model.train()
    ims, ce_masks = data
    ims = ims.to(config.DEVICE)
    ce_masks = ce_masks.to(config.DEVICE)
    _masks = model(ims)
    optimizer.zero_grad()
    loss, acc = criterion(_masks, ce_masks)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()

  @torch.no_grad()
  def validate_batch(model, data, criterion):
    model.eval()
    with torch.no_grad():
        ims, masks = data
        ims = ims.to(config.DEVICE)
        masks = masks.to(config.DEVICE)
        _masks = model(ims)

        loss, acc = criterion(_masks, masks)

        return loss.item(), acc.item()