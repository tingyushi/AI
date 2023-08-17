import numpy as np
import torch
import collections
collections.Iterable = collections.abc.Iterable

from dataloader import MnistDataloader
from vit import VIT, train
from hyperparameters import *


# load data
dataloader = MnistDataloader(training_images_filepath='data/train_images', 
                                                      training_labels_filepath='data/train_labels',
                                                      test_images_filepath='data/test_images',
                                                      test_labels_filepath='data/test_labels')
(x_train, y_train),(x_test, y_test) = dataloader.load_data()

'''
preprocess data
train shape (number of pictures, height, weight, channel)
label shape (number of pictures)
'''
x_train = np.array(x_train) ; x_train = torch.tensor(x_train, dtype=torch.float) ; x_train = torch.unsqueeze(x_train, -1)
y_train = np.array(y_train) ; y_train = torch.tensor(y_train, dtype=torch.int8)
x_test = np.array(x_test) ; x_test = torch.tensor(x_test, dtype=torch.float) ; x_test = torch.unsqueeze(x_test, -1)
y_test = np.array(y_test) ; y_test = torch.tensor(y_test, dtype=torch.int8)


'''
# get a smaller size data 
x_train = x_train[:50] ; y_train = y_train[:50]
x_test = x_test[:50] ; y_test = y_test[:50]
'''

# create model
model = VIT(token_dim=PATCH_DIM, 
            n_head=N_HEAD,
            n_block=N_BLOCK,
            patch_size=PATCH_SIZE,
            height=HEIGHT,
            width= WIDTH,
            channel=CHANNEL,
            n_class=N_CLASS,
            device=DEVICE)


# move to gpu
model = model.to(DEVICE)
x_train = x_train.to(DEVICE) ; y_train = y_train.to(DEVICE)
x_test = x_test.to(DEVICE) ; y_test = y_test.to(DEVICE)

print()
print(f"Training on: {DEVICE}")
print()

# train
train(model=model,
      x_train=x_train,
      y_train=y_train,
      x_test=x_test,
      y_test=y_test,
      epoch=EPOCH,
      lr=LR)
