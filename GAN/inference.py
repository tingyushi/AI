from model import get_generator
from hparams import LATENT_DIM
import tensorflow as tf
import matplotlib.pyplot as plt


# load weights
gen = get_generator(input_dim=LATENT_DIM)
gen.load_weights('trained_model/generator.h5')

# make predictions
img = gen.predict(tf.random.normal((1, 128, 1)))
img = img[0]

# save fig
plt.imshow(img, 'gray')
plt.title('Generated Image')
plt.savefig('images/generated_image.png')