import torch
class config:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    N_EPOCHS = 100
    BATCH_SIZE = 4
    DICE_COEF = 1
    LOVASZ_COEF = 0.3