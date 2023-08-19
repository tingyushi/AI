import torch
import collections
collections.Iterable = collections.abc.Iterable
from dataloader import MnistDataloader
import sys
from hyperparameters import *
from vit import VIT
import numpy as np
import matplotlib.pyplot as plt 


# load data
dataloader = MnistDataloader(training_images_filepath='data/train_images', 
                                                      training_labels_filepath='data/train_labels',
                                                      test_images_filepath='data/test_images',
                                                      test_labels_filepath='data/test_labels')
(_, _),(x_test, y_test) = dataloader.load_data()
x_test = np.array(x_test) ; x_test = torch.tensor(x_test, dtype=torch.float) ; x_test = torch.unsqueeze(x_test, -1)
y_test = np.array(y_test) ; y_test = torch.tensor(y_test, dtype=torch.int8)
x_test = x_test.to(DEVICE) ; y_test = y_test.to(DEVICE)

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


# load parameters
model.load_state_dict(torch.load('trained_model/model'))
model.to(DEVICE)

# predict
idx = eval(sys.argv[1])
img = x_test[idx]; label = y_test[idx].item() ; pred_label = model.predict(img)
plt.imshow(img.cpu(), 'gray')
plt.title(f"Predicted Label: {pred_label} -- True Label: {label}")
plt.savefig('example_prediction.png')
