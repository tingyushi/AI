'''
This file contains functions to prepreocess images
'''

from PIL import Image
import numpy as np


# convert image size
def preprocess_image(img_path, model_image_size):

    image = Image.open(img_path)
    # resize image
    resized_image = image.resize(tuple(reversed(model_image_size)), 
                                 Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0) 
    return image, image_data

