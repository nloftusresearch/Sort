import sys
sys.path.append('/remote_home/Thesis')

import argparse
import tensorflow as tf
import numpy as np
from utils.coco_dataset_manager import *
import xml.etree.ElementTree as ET
import tensorflow as tf
import keras_cv
from utils.yolo_utils import *
from utils.custom_retinanet import prepare_image
from utils.nonmaxsuppression import PreBayesianNMS
from pycocotools.coco import COCO
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import os
import ast
import json
import pickle


# Path to .txt files with classes to be included
cls_path = "/remote_home/Thesis/Prebayesian/yolo-cls-list.txt"

# Path to where checkpoint for current best model is saved
checkpoint_path = "/remote_home/Thesis/Prebayesian/model_training/.5_LR_2x_CF_weightsonly"

# Path to folder with images
image_directory = "/remote_home/Thesis/BDD_Files/traffic"

# Path for detections to be temporarily saved
file_path = '/remote_home/Thesis/Prebayesian/frames_output.txt'

# Define the output path for the parsed dictionary of objects found in frame
dict_output_path = '/remote_home/Thesis/Sort/parsed_data_dict.pkl'


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs," , len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

max_iou = .2
min_confidence = .1

#Load the class lists from text, if not specified, it gets all 80 classes
if (cls_path == ""):
    cls_list = None
else:
    with open(cls_path) as f:
        cls_list = f.readlines()
        cls_list = [cls.replace("\n", "") for cls in cls_list]
num_classes = 80 if cls_list is None else len(cls_list)


nms = PreBayesianNMS("xywh", True, confidence_threshold=min_confidence)

backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_s_backbone_coco"  # We will use yolov8 small backbone with coco weights
)

model = keras_cv.models.YOLOV8Detector(
    num_classes= num_classes,
    bounding_box_format="xywh",
    backbone=backbone,
    fpn_depth=2,
    prediction_decoder=nms
)


latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
model.load_weights(latest_checkpoint).expect_partial()

def load_images(image_dir):
    images = []

    # List all files in the directory
    file_names = os.listdir(image_dir)

    # Sort file names in ascending order
    file_names.sort()

    for filename in file_names:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image)
            images.append(image)

    return images

# Load images from a directory
print("loading images")
images = load_images(image_directory)
print("loaded images")

# Open the file in write mode

with open(file_path, 'w') as file:
    for frame_number, img in enumerate(images):
        try:
            image = tf.cast(img, dtype=tf.float32)
            input_image, ratio = prepare_image(image)
            detections = model.predict(input_image)
            

            boxes = np.asarray(detections["boxes"][0])
            cls_prob = np.asarray(detections["cls_prob"])
            cls_id = np.asarray(detections["cls_idx"][0])
            cls_name = [cls_list[x] for x in cls_id]

            correct_prob = []
            for i in range(len(cls_prob)):
                correct_prob.append(cls_prob[i][cls_id[i]])

            # Write frame information to the file
            file.write(f'{frame_number}:{[boxes, cls_prob, cls_id, cls_name]}\n')

            # Commented out for now since it might cause memory issues
            # visualize_detections(image, boxes, cls_name, correct_prob)
        except IndexError:
            print(f"No valid detections for Frame {frame_number}")
        

print(f'Output has been written to {file_path}')


def read_valuestring(valstring):
    valstring = valstring[1:-1]
    valstring = valstring.replace("\n", " ")
    valstring = valstring.replace("array", "")
    valstring = valstring.replace("dtype=float32", "")
    valstring = ast.literal_eval(valstring)

    valstring = {
        'boxes': valstring[0][0],
        'probabilities': valstring[1][0][0],
        'classes': valstring[2],
        'names': valstring[3]
    }
    
    return valstring

# Read data from the file
with open(file_path, 'r') as file:
    data_lines = file.readlines()

# Create a dictionary to store the parsed data
data_dict = {}

# Initialize variables to store key and value
current_key = None
current_value_lines = []

# Iterate through each line
for line in data_lines:
    # Check if the line contains ":"
    if ":" in line:
        # If a key is already set, save the previous key-value pair
        if current_key is not None:
            # Combine the lines into a single string
            value_str = ''.join(current_value_lines).strip()

            data_dict[int(current_key)] = read_valuestring(value_str)

        # Split the line into key and value using the first occurrence of ":"
        current_key, current_value_lines = line.split(':', 1)
        # Reset value for the new key
        current_value_lines = [current_value_lines]
    else:
        # If no ":" is found, add the line to the current value
        current_value_lines.append(line)

    
# Add the last key-value pair to the dictionary
if current_key is not None:
    # Combine the lines into a single string
    value_str = ''.join(current_value_lines).strip()

    # Assign the key-value pair to the dictionary
    data_dict[int(current_key)] = read_valuestring(value_str)


# Save the dictionary to a pickle file using pickle.dump
with open(dict_output_path, 'wb') as pickle_file:
    pickle.dump(data_dict, pickle_file)

print(f"Dictionary saved to {dict_output_path}")

# Load data from a pickle file
with open(dict_output_path, 'rb') as pickle_file:
    loaded_frames_detections = pickle.load(pickle_file)

