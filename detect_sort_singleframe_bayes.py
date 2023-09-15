import os
import sort
from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np
import random
from motmetrics import metrics, utils  # Import motmetrics modules
import csv
from sklearn.naive_bayes import GaussianNB


print("Detect and Sort with Naive Bayes")

def naive_bayes(track_bbs_ids, confs, massive_object_dictionary):
    """
    Implement naive bayes with class predictions and their associated confidences
    to make a final class prediction with an associated confidence.

    Args:
      track_bbs_ids: A list of track ids.
      confs: A list of confidences.

    Returns:
      A dictionary of object ID to a tuple of (final_class_prediction, final_confidence).
    """
    
    # Create a dictionary for current frame id predictions
    bayes_predictions = {}
    
    # For each bounding box
    for (xmin, ymin, xmax, ymax, obj_id), (confidence, class_index) in zip(track_bbs_ids, confs):
        # If the object has been seen already
        if obj_id in massive_object_dictionary.keys():
            # Append current confidence and class index
            massive_object_dictionary[obj_id][0].append(float(confidence))
            massive_object_dictionary[obj_id][1].append(int(class_index))
        # Otherwise create a new dict index for the id
        else:
            massive_object_dictionary[obj_id] = [[float(confidence)], [int(class_index)]]

    # Now that we have processed all bounding boxes for this frame,
    # calculate final predictions and add them to the bayes_predictions dictionary
    for obj_id, (confidences, class_indices) in massive_object_dictionary.items():
        class_predictions = class_indices
        num_classes = len(set(class_predictions))
        prior_probabilities = np.ones(num_classes) / num_classes
        conditional_probabilities = {}

        for i in range(num_classes):
            prior_probabilities[i] = len([j for j in class_predictions if j == i]) / len(class_predictions)

        for i in range(num_classes):
            conditional_probabilities[i] = {}
            for j in range(num_classes):
                count = 0
                total = 0
                for _, (confidences, class_indices) in massive_object_dictionary.items():
                    if i in class_indices and j in class_indices:
                        count += 1
                    if i in class_indices:
                        total += 1

                if total > 0:
                    conditional_probabilities[i][j] = count / total

        final_class_prediction = np.argmax(prior_probabilities)
        final_confidence = prior_probabilities[final_class_prediction]
        
        # Add values to the bayes_predictions dict
        bayes_predictions[obj_id] = final_class_prediction, final_confidence
    print(bayes_predictions)
    return bayes_predictions





# Define a function to generate random colors
def generate_random_colors(num_colors):
    random.seed(42)  # Set a seed for reproducibility
    colors = []
    for _ in range(num_colors):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        colors.append((r, g, b))
    return colors


# Read the raw text from the file
with open(r'D:\Coding\Thesis\sort\object_dict.txt', 'r') as file:
    raw_text = file.read()

# Find the part of the text that represents the dictionary (remove variable name)
start_index = raw_text.find('{')
end_index = raw_text.rfind('}')
dictionary_text = raw_text[start_index:end_index + 1]

# Use eval to convert the dictionary text into a dictionary
color_mapping = eval(dictionary_text)
    
colors = generate_random_colors(len(color_mapping))  # Generate random colors based on the number of classes

# Initialize SORT tracker
mot_tracker = sort.Sort()

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Path to the image folder
image_folder = Path(r"D:\Coding\Thesis\MOT17\train\MOT17-13-DPM\img1")

# Get the dimensions of the first image in the folder
first_image = next(image_folder.iterdir())
first_image = cv2.imread(str(first_image))
height, width, _ = first_image.shape

# Define the output video path
output_video_path = 'output_video_single_bayes.avi'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))


# Create a dictionary to store the class predictions and their associated confidences for each track id.
massive_object_dictionary = {}


# Create and open a CSV file for writing tracking results
with open('detect_and_sort_results.csv', mode='w', newline='') as csv_file:
    fieldnames = ['FrameNumber', 'ObjectID', 'X', 'Y', 'Width', 'Height', 'Confidence', 'Class']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # For each image
    for image_path in image_folder.iterdir():
        image = cv2.imread(str(image_path))
        results = model.predict(image)[0]
        
        detections = []
        boxes = results.boxes
        confs = boxes.data[:, 4:6]
        classes = []
        
        # Get all of the box data
        for box, conf in zip(boxes, confs):
            b = box.xyxy[0].tolist()
            c = conf.tolist()
            b.append(c[0])
            detections.append(b)
                
        # And update the tracker
        track_bbs_ids = mot_tracker.update(detections)
       
        # Using the updated tracker 
        frame_id_predictions = naive_bayes(track_bbs_ids, confs, massive_object_dictionary)

        for obj_id, (xmin, ymin, xmax, ymax, class_index, confidence) in frame_id_predictions.items():

            # Process values
            class_index = int(class_index) + 1
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            class_name = f"Class {color_mapping[class_index]}"  # Cast class_index to int for class name

            # Add box data to frame
            object_id_and_class = f"{class_name}: {confidence:.3f}"  # Concatenate Object ID and Confidence
            color = colors[int(class_index)] if class_index < len(colors) else (0, 0, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, object_id_and_class, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
            # Write the tracking result to the CSV file
            writer.writerow({
                'FrameNumber': image_path.stem,
                'ObjectID': int(obj_id),
                'X': xmin,
                'Y': ymin,
                'Width': xmax - xmin,
                'Height': ymax - ymin,
                'Confidence': float(confidence),
                'Class': class_index
            })

        # Write the frame with bounding boxes to the output video
        out.write(image)

# Release the VideoWriter object and close the video file
out.release()


print(f'Video saved as {output_video_path}')















'''
from ultralytics import YOLO

# Load a model
model = YOLO(r"C:\PathToYourModel\model.pt")

# Predict a class of an image
results = model(r"C:\PathToYourImage\image.jpg")

# Print the result
if results[0].probs[0] > results[0].probs[1]:
    print("First class")
else:
    print("Second class")
'''