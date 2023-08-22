from hparams import *
import tensorflow as tf
import numpy as np
from model import get_generator, get_discriminator, GAN
from tensorflow.keras.optimizers.legacy import Adam 
from tensorflow.keras.losses import BinaryCrossentropy
from progressbar import progressbar
import matplotlib.pyplot as plt


# load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train
# gather together all the images
x = np.concatenate([x_train, x_test], 0) ; y = np.concatenate([y_train, y_test], 0)

# add channel dimension
x = np.expand_dims(x, axis = -1)

# scale data
x = x.astype('float64')
y = y.astype('uint8')
x /= 255.0

# make x and y to batches
# x: (number of batches, batch size, height, width, channel)
# y: (number of batches, batch size, 1)
num_of_pics = x.shape[0] ; num_of_batches = int(num_of_pics / BATCH_SIZE)
x_batch = np.zeros((num_of_batches, BATCH_SIZE, HEIGHT, WIDTH, CHANNEL), dtype = np.float64)
y_batch = np.zeros((num_of_batches, BATCH_SIZE), dtype = np.uint8)
for i in range(num_of_batches):
    x_batch[i] = x[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    y_batch[i] = y[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]


# convert to tensor
x = tf.convert_to_tensor(x)
x_batch = tf.convert_to_tensor(x_batch)
y = tf.convert_to_tensor(y)
y_batch = tf.convert_to_tensor(y_batch)

print("Data Shapes")
print(f"x_batch shape: {x_batch.shape}") 
print(f"y_batch shape: {y_batch.shape}")
print(f"x shape: {x.shape}") 
print(f"y shape: {y.shape}")

print(f"x_batch dtype: {x_batch.dtype}") 
print(f"y_batch dtype: {y_batch.dtype}")
print(f"x dtype: {x.dtype}") 
print(f"y dtype: {y.dtype}")

print()


# generator and discriminator
gen = get_generator(input_dim = LATENT_DIM)
dis = get_discriminator((HEIGHT, WIDTH, CHANNEL))

# define loss 
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()

# define optimizers
g_opt = Adam(learning_rate = 0.0001)
d_opt = Adam(learning_rate = 0.00001)

gan_model = GAN(gen, dis, LATENT_DIM)
gan_model.compile(g_opt, d_opt, g_loss, d_loss)

d_losses = []
g_losses = []

for epoch in range(EPOCHS):
    print(f"Epoch: {epoch}")
    for batch_idx in progressbar( range(num_of_batches) ):
        losses = gan_model.train_step(x_batch[batch_idx])
        d_losses.append(losses['dis_loss'])
        g_losses.append(losses['gen_loss'])
        if batch_idx == (num_of_batches - 1):
            print(f"Discriminator Loss: {losses['dis_loss']} --- Generator Loss: {losses['gen_loss']}")
    print()


# plot loss
plt.plot(d_losses, label='d_loss')
plt.plot(g_losses, label='g_loss')
plt.title('Loss')
plt.legend()
plt.savefig('images/Loss.png')

# save model
gen.save('trained_model/generator.h5')