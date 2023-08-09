import torch
from torchvision.models import vgg16_bn
import torchvision
import torch.nn as nn
from .loss import SegmentationLoss
from .config import config
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import segmentation_models_pytorch as smp

def make_model():
    model = smp.FPN(
    encoder_name="timm-efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
    ).to(config.DEVICE)

    criterion = SegmentationLoss
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, weight_decay = config.WEIGHT_DECAY, momentum=config.MOMENTUM, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.N_EPOCHS, eta_min= 1e-7)

    return model, criterion, optimizer, scheduler