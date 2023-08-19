import numpy as np
import torch
import collections
collections.Iterable = collections.abc.Iterable
from dataloader import MnistDataloader
from vit import VIT, train, parameters
from progressbar import progressbar
from hyperparameters import *
import random

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


# create model
model = VIT(token_dim=TOKEN_DIM, 
            n_head=N_HEAD,
            n_block=N_BLOCK,
            patch_size=PATCH_SIZE,
            height=HEIGHT,
            width= WIDTH,
            channel=CHANNEL,
            n_class=N_CLASS,
            device=DEVICE)

print()
print(f"Training on: {DEVICE}")
print()
print(f"Number of parameters: {parameters(model)}")
print()

'''
The following code on select a subset of train and test set
'''
data_size = 400
train_idx = random.sample(range(x_train.shape[0]), data_size)
test_idx = random.sample(range(x_test.shape[0]), int(data_size * 0.8))
xtrain = torch.zeros((data_size, HEIGHT, WIDTH, CHANNEL), dtype=torch.float)
ytrain = torch.zeros((data_size), dtype=torch.int8)
xtest = torch.zeros((int(data_size * 0.8), HEIGHT, WIDTH, CHANNEL), dtype=torch.float)
ytest = torch.zeros((int(data_size * 0.8)), dtype=torch.int8)

for i in range(data_size):
    xtrain[i] = x_train[i]
    ytrain[i] = y_train[i]

for i in range(int(data_size * 0.8)):
    xtest[i] = x_test[i]
    ytest[i] = y_test[i]

# move to gpu
model = model.to(DEVICE)
xtrain = xtrain.to(DEVICE) ; ytrain = ytrain.to(DEVICE)
xtest = xtest.to(DEVICE) ; ytest = ytest.to(DEVICE)


# uncomment the following to train on whole data set
train(model=model,
      x_train=xtrain,
      y_train=ytrain,
      x_test=xtest,
      y_test=ytest,
      epoch=EPOCH,
      lr=LR)
