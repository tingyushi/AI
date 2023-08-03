'''
Neural Style Transfer
'''
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
from progressbar import progressbar


def load_model(parameter_path):
    model = tf.keras.applications.VGG19(include_top = False,
                                  input_shape = IMAGE_SHAPE,
                                  weights = parameter_path)
    
    for layer in model.layers:
        layer.trainable = False

    return model



# get layer outputs from layer_names
def get_layer_outputs(model, layer_names):
    outputs = [model.get_layer(layer[0]).output for layer in layer_names]
    model = tf.keras.Model([model.input], outputs)
    return model    



# generate initial final image
def initial_generated_image(content_image):
    generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
    generated_image = tf.add(generated_image, noise)
    generated_image = tf.clip_by_value(generated_image, 
                                       clip_value_min=0.0, 
                                       clip_value_max=1.0)
    return tf.Variable(generated_image)



# read an image and store in tensor
def read_image(image_path, image_size):
    image = np.array(Image.open(image_path).resize((image_size)))
    image = tf.constant(np.reshape(image, ((1,) + image.shape)))
    return image



def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), 
    hidden layer activations representing content of the image C
    
    a_G -- tensor of dimension (1, n_H, n_W, n_C), 
    hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- content cost
    """
    
    a_G = a_G[-1]
    
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape 'a_C' and 'a_G' (≈2 lines)
    a_C_unrolled = tf.reshape(a_C, shape = [ -1 , n_C])
    a_G_unrolled = tf.reshape(a_G, shape = [ -1 , n_C])
    
    # compute cost
    J_content = tf.subtract(a_C_unrolled, a_G_unrolled)
    J_content = tf.square(J_content)
    J_content = tf.reduce_sum(J_content)
    J_content /= (4 * n_H * n_W * n_C)
        
    return J_content



def gram_matrix(A):
    return tf.linalg.matmul(A, tf.transpose(A))



def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), 
    hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), 
    hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value
    """
    
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_S = tf.transpose(tf.reshape(a_S, [n_H * n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = tf.reduce_sum( tf.square( tf.subtract(GS, GG) ) )
    J_style_layer /= (4 * (n_C**2) * ((n_W*n_H)**2))
    
    return J_style_layer


def compute_style_cost(a_S, a_G, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    """
    
    J_style = 0
    
    a_G = a_G[:-1]
    
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):  
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])
        J_style += weight[1] * J_style_layer

    return J_style



#Truncate all the pixels in the tensor to be between 0 and 1
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)




def tensor_to_image(tensor):
    """
    Converts the given tensor into a PIL image
    
    Arguments:
    tensor -- Tensor
    
    Returns:
    Image: A PIL image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)




@tf.function()
def total_cost(J_content, J_style, alpha, beta):
    return alpha * J_content + beta * J_style




@tf.function()
def train_step(generated_image):
    with tf.GradientTape() as tape:
        a_G = all_layers_outputs(generated_image)
        J_style = compute_style_cost(a_S, a_G, STYLE_LAYERS = STYLE_LAYERS)
        J_content = compute_content_cost(a_C, a_G)
        J = total_cost(J_content, J_style, alpha = ALPHA, beta = BETA) 
    grad = tape.gradient(J, generated_image)
    OPTIMIZER.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))
    return J


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("WRONG NUMBER OF ARGUMENTS")
        exit()

    # modify the following two variables to change content and style image
    CONTENT_IMAGE_NAME = sys.argv[1]
    STYLE_IMAGE_NAME = sys.argv[2]
    OUPUT_IMAGE_NAME = sys.argv[3]


    # define other constants
    IMAGE_SIZE = (400, 400) ; IMAGE_SHAPE = (400, 400, 3)
    INPUT_IMAGE_PATH = 'input_images/'
    OUTPUT_IMAGE_PATH = 'output_images/'
    MODEL_PARAM_PATH = 'vgg19_weights.h5'
    STYLE_LAYERS = [('block1_conv1', 0.2),
                    ('block2_conv1', 0.2),
                    ('block3_conv1', 0.2),
                    ('block4_conv1', 0.2),
                    ('block5_conv1', 0.2)]
    CONTENT_LAYER = [('block5_conv4', 1)]
    ALPHA = 10 ; BETA = 40
    EPOCHS = 2500
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.001)

    if not CONTENT_IMAGE_NAME in os.listdir(INPUT_IMAGE_PATH):
        print("WRONG INPUT IMAGE NAME")
        exit()

    if not STYLE_IMAGE_NAME in os.listdir(INPUT_IMAGE_PATH):
        print("WRONG STYLE IMAGE NAME")
        exit()
    

    '''
    for f in os.listdir(OUTPUT_IMAGE_PATH):
        os.remove(os.path.join(OUTPUT_IMAGE_PATH, f))
    '''

    # load model
    model = load_model(MODEL_PARAM_PATH)
    
    # read images
    content_image = read_image(INPUT_IMAGE_PATH + CONTENT_IMAGE_NAME, IMAGE_SIZE)
    style_image = read_image(INPUT_IMAGE_PATH + STYLE_IMAGE_NAME, IMAGE_SIZE)
    
    # initial generated image
    generated_image = initial_generated_image(content_image)

    # get layer outputs
    content_layer_output = get_layer_outputs(model, CONTENT_LAYER)
    style_layers_outputs = get_layer_outputs(model, STYLE_LAYERS)
    all_layers_outputs = get_layer_outputs(model, STYLE_LAYERS + CONTENT_LAYER)
    
    #get a_C and a_S
    a_C =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    a_C = content_layer_output(a_C)
    a_S =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
    a_S = style_layers_outputs(a_S)

    # train
    for i in progressbar( range(EPOCHS) ):
        train_step(generated_image)

    generated_image = tensor_to_image(generated_image)
    generated_image.save(OUTPUT_IMAGE_PATH + OUPUT_IMAGE_NAME)