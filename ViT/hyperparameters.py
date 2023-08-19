import torch


# defind hyperparameters
PATCH_SIZE = 4
HEIGHT = 28
WIDTH = 28
CHANNEL = 1
NUM_OF_PATCH = int((HEIGHT / PATCH_SIZE) ** 2) 
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * CHANNEL
N_HEAD = 2 # number of heads in multi-head attention
HEAD_SIZE = int(PATCH_DIM / N_HEAD)
N_BLOCK = 2
N_CLASS = 10
EPOCH = 200
LR = 1e-2
TOKEN_DIM = int(PATCH_DIM / 2)
DEVICE = 'cpu'
