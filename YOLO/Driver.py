import preprocessor
import model
import postprocessor
import os
import sys
from progressbar import progressbar
import skvideo.io
import matplotlib.pyplot as plt
import numpy as np

def handle_one_picture(input_image_path, 
                       image_size,
                       model_path,
                       anchor_path,
                       classes_path):
    # preprocess image
    image, _ = preprocessor.preprocess_image(img_path = input_image_path,
                                             model_image_size = image_size)

    # LOAD MODEL
    yolo_model = model.load_model(model_path)

    # read anchors
    anchors = model.read_anchors(anchor_path)

    # read classes
    classes = model.read_classes(classes_path)

    # make predictions
    out_scores, out_boxes, out_classes = postprocessor.predict(image_file_path = input_image_path,
                                                               image_size = image_size, 
                                                               yolo_model = yolo_model, 
                                                               anchors = anchors,
                                                               classes = classes)
    
    # draw boxes
    postprocessor.draw_boxes(image = image, 
                             boxes = out_boxes,
                             box_classes = out_classes, 
                             class_names = classes,
                             scores = out_scores)
    
    return image


if __name__ == '__main__':
    IMAGE_SIZE = (608, 608)
    INPUT_IMAGE_PATH = 'input_images/' 
    OUTPUT_IMAGE_PATH = 'output_images/' 
    OUTPUT_VIDEO_PATH = 'output_video/'
    MODEL_PATH = 'model_data/'
    ANCHOR_PATH = 'model_data/yolo_anchors.txt'
    CLASSES_PATH = 'model_data/coco_classes.txt'

    
    # remove ouput images
    for f in os.listdir(OUTPUT_IMAGE_PATH):
        if not f.endswith('.md'):
            os.remove(os.path.join(OUTPUT_IMAGE_PATH, f))
    
    # remove output video
    for f in os.listdir(OUTPUT_VIDEO_PATH):
        if not f.endswith('.md'):
            os.remove(os.path.join(OUTPUT_VIDEO_PATH, f))
    
    # collect input image names
    INPUT_IMAGE_NAMES = []
    for image_name in os.listdir(INPUT_IMAGE_PATH):
        if not image_name.startswith('.'):
            INPUT_IMAGE_NAMES.append(image_name)
    INPUT_IMAGE_NAMES.sort()
    
    for idx , image_name in progressbar( enumerate(INPUT_IMAGE_NAMES) ):
        annotated_image = handle_one_picture(input_image_path = INPUT_IMAGE_PATH + image_name,
                                                image_size =  IMAGE_SIZE,
                                                model_path = MODEL_PATH,
                                                anchor_path = ANCHOR_PATH,
                                                classes_path =  CLASSES_PATH)
        output_path = OUTPUT_IMAGE_PATH + str(idx + 1) + '.jpg'
        annotated_image.save(output_path)
    

    # generate video
    fps = 4
    frames = []
    images = []
    for image in os.listdir(OUTPUT_IMAGE_PATH):
        if (not image.startswith('.')) and (not image.endswith('.md')):
            images.append(image)
    images.sort()

    for image in images:        
        frames.append(plt.imread(os.path.join(OUTPUT_IMAGE_PATH + image)))
    skvideo.io.vwrite(OUTPUT_VIDEO_PATH + 'video.mp4', np.array(frames), 
                      inputdict={'-r': str(fps)})
