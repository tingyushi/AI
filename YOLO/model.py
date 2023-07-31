'''
This file contains functions to 
    1. load model
    2. make predictions
    3. decode predictions
'''

from tensorflow.keras.models import load_model as lm
from tensorflow.keras import backend as K
import numpy as np


def load_model(model_path, compile = False):
    model = lm(model_path, compile = compile)
    return model


def yolo_head(feats, anchors, class_names):
    """Convert final layer features to bounding box parameters.

    Parameters
    ----------
    feats : tensor
        Final convolutional layer features.
    anchors : array-like
        Anchor box widths and heights.
    class_names : array-like
        class names of 80 classes

    Returns
    -------
    box_xy : tensor
        x, y box predictions adjusted by spatial location in conv layer.
    box_wh : tensor
        w, h box predictions adjusted by anchors and conv spatial resolution.
    box_conf : tensor
        Probability estimate for whether each box contains any object.
    box_class_pred : tensor
        Probability distribution estimate for each box over class labels.
    """

    num_anchors = len(anchors)
    num_classes = len(class_names)
    
    anchors_tensor = K.reshape(K.variable(anchors), 
                               [1, 1, 1, num_anchors, 2])

    
    conv_dims = K.shape(feats)[1:3]  
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])
    conv_width_index = K.tile(K.expand_dims(conv_width_index, 0), 
                              [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, 
                                      conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], 
                                        conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))
    '''
    The inner dimention has something like 
    [0, 0], [1, 0], [2, 0] ... [18, 18]
    '''

    feats = K.reshape(feats, 
                      [-1, conv_dims[0], 
                       conv_dims[1], 
                       num_anchors, 
                       num_classes + 5])
    
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), 
                       K.dtype(feats))


    # shape (m , 19, 19, 5, 2)
    box_xy = K.sigmoid(feats[..., :2])

    # shape (m , 19, 19, 5, 2)
    box_wh = K.exp(feats[..., 2:4])

    #shape (m , 19, 19, 5, 1)
    box_confidence = K.sigmoid(feats[..., 4:5])

    #shape (m, 19, 19, 5, 80)
    box_class_probs = K.softmax(feats[..., 5:])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_xy, box_wh, box_confidence, box_class_probs


def read_classes(file_path):
    with open(file_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(file_path):
    with open(file_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def generate_output(model, image_data, anchors, classes):
    model_output = model(image_data)

    # decode model output 
    output = yolo_head(model_output, anchors, classes)

    return output