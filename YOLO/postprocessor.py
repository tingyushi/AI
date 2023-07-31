'''
Decode outputs from the mode
'''

import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from PIL import ImageDraw, ImageFont

from preprocessor import preprocess_image
from model import generate_output

def yolo_filter_boxes(boxes, 
                      box_confidence, 
                      box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
        boxes -- tensor of shape (19, 19, 5, 4) (bx, by, bh, bw)
        box_confidence -- tensor of shape (19, 19, 5, 1) (pc)
        box_class_probs -- tensor of shape (19, 19, 5, 80) (c1 -> c80)
        threshold -- real value, if [ highest class probability score < threshold],
                     then get rid of the corresponding box

    Returns:
        scores -- tensor of shape (None,), containing the class probability score for selected boxes
        boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    
    """
    
    # shape = (19, 19, 5, 80)
    box_scores = box_confidence * box_class_probs


    # shape = (19, 19, 5, 1)
    box_classes = tf.math.argmax(box_scores, 
                                 axis = -1)
    
    # shape = (19, 19, 5, 1)
    box_class_scores = tf.math.reduce_max(box_scores, 
                                          axis = -1, 
                                          keepdims = False)
    
    # create mask
    # shape (19, 19, 5, 1)
    filtering_mask = tf.zeros(box_class_scores.shape)
    filtering_mask += threshold
    filtering_mask = box_class_scores > filtering_mask
    
   
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
 
    return scores, boxes, classes


# apply NMS to reduce number of boxes
def yolo_non_max_suppression(scores, 
                             boxes, 
                             classes, 
                             max_boxes = 10, 
                             iou_threshold = 0.5):

    # maximum number of boxes in one image
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')

    nms_indices = tf.image.non_max_suppression(boxes = boxes, 
                                               scores = scores,
                                               max_output_size = max_boxes_tensor,
                                               iou_threshold = iou_threshold)
    

    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
    
    return scores, boxes, classes


# Scales the predicted boxes in order to be drawable on the image
def scale_boxes(boxes, image_shape):
    height = float(image_shape[0])
    width = float(image_shape[1])
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

#Convert YOLO box predictions to bounding box corners 
def yolo_boxes_to_corners(box_xy, box_wh):
   
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],  # y0
        box_mins[..., 0:1],  # x0
        box_maxes[..., 1:2],  # y1
        box_maxes[..., 0:1]  # x1
    ])


def yolo_eval(yolo_outputs, 
              image_shape, 
              max_boxes=10, 
              score_threshold=.6, 
              iou_threshold=.5):
    """
    Generate information for boxes to be shown in the image
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)

    image_shape -- tensor of shape (2,) containing the input shape
    max_boxes --  max number of boxes 
    score_threshold -- used for filtering boxes 
    iou_threshold -- used for NMS

    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    
    '''
    Retrive model outpus

    box_xy: shape (m, 19, 19, 5, 2)
    box_wh: shape (m, 19, 19, 5, 2)
    box_confidence: shape (m , 19, 19, 5, 1)
    box_class_probs: shape (m , 19, 19, 5, 80)
    '''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    
    # get boxes corners information
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    
    
    '''
    filter boxes with low probabilities

    scores: shape (None, )
    boxes: shape (None, 4)
    classes: shape (None, )
    '''
    scores, boxes, classes = yolo_filter_boxes(boxes,
                                               box_confidence,
                                               box_class_probs,
                                               score_threshold)
    
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)
    
    # NMS
    scores, boxes, classes = yolo_non_max_suppression(scores, 
                             boxes, 
                             classes, 
                             max_boxes = max_boxes, 
                             iou_threshold = iou_threshold)
    
    return scores, boxes, classes


def predict(image_file_path, image_size, yolo_model, anchors, classes):
    """
    Arguments:
    image_file_path -- path of the input image
    image_size: size of image used for model
    yolo_model: model

    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes 
    """

    # Preprocess your image
    image, image_data = preprocess_image(image_file_path, model_image_size = image_size)
    
    # make predictions
    yolo_outputs = generate_output(yolo_model, image_data, anchors, classes)

    out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.size[1],  image.size[0]], 10, 0.3, 0.5)

    return out_scores, out_boxes, out_classes



# draw bounding boxes on image
def draw_boxes(image, boxes, box_classes, class_names, scores=None):
    
    ORANGE = '#FFA500'
    RED = '#F70100'
    
    for i, c in list(enumerate(box_classes)):
        
        # retrive class name and box
        box_class = class_names[c]
        box = boxes[i]
        
        if isinstance(scores.numpy(), np.ndarray):
            score = scores.numpy()[i]
            label = '{} {:.2f}'.format(box_class, score)
        else:
            label = '{}'.format(box_class)

        draw = ImageDraw.Draw(image)

        # upper left (x0, y0)
        # bottom right (x1, y1)
        y0, x0, y1, x1 = box
        y0 = max(0, np.floor(y0 + 0.5).astype('int32'))
        x0 = max(0, np.floor(x0 + 0.5).astype('int32'))
        y1 = min(image.size[1], np.floor(y1 + 0.5).astype('int32'))
        x1 = min(image.size[0], np.floor(x1 + 0.5).astype('int32'))
        
        height = abs(y1 - y0) ; width = abs(x1 - x0)

        # draw rectangle
        draw.rectangle([x0, y0+10, x1, y1], width = 3, outline = ORANGE)

        # define font
        size = max(15, np.uint32(height / 10) )
        font = ImageFont.truetype(font = 'font/FiraMono-Medium.otf',
                                  size = size)
        draw.text((x0, y0 - 0.7*size), label, RED , font=font)

        del draw