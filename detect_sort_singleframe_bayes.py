import os
import sort
from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np
import random
from motmetrics import metrics, utils  # Import motmetrics modules
import csv
from scipy.stats import beta


print("Detect and Sort with Naive Bayes")


def naive_bayes(obj_id, confidence, clas, massive_object_dictionary):
    """Calculates the overarching prediction for the object class and returns the associated confidence.

    Args:
    obj_id: The integer id of the object in question
    confidence: Confidence in current class prediction
    clas: Current class prediction
    massive_object_dictionary: All previous predictions for all ids

    Returns:
    A tuple containing the predicted object class and the associated confidence as well as updated massive_object_dictionary
    """

    # Extract the previous predicted classes and confidences for the given obj_id
    prev_predicted_classes = massive_object_dictionary[obj_id][1]
    prev_confidences = massive_object_dictionary[obj_id][0]


    # Add the current predicted class and confidence to the previous predicted classes and confidences
    prev_predicted_classes = np.append(prev_predicted_classes, clas)
    prev_confidences = np.append(prev_confidences, confidence)

    
    # Get array of counts for each class
    class_counts = np.unique(prev_predicted_classes, return_counts=True)[1]
    # Normalize the values
    prior_probabilities = class_counts / np.sum(class_counts)

    # print(class_counts)
    # print(prior_probabilities)

    # Calculate the likelihood of each class prediction given the current class prediction.
    class_likelihoods = np.zeros((len(prev_confidences), len(prior_probabilities)))
    for i, predicted_class in enumerate(prev_predicted_classes):
        for j, object_class in enumerate(prior_probabilities):
            if predicted_class == object_class:
                class_likelihoods[i, j] += prev_confidences[i]
            
    # print(class_likelihoods)

    # Apply Bayes' theorem to calculate the posterior probability of each class.
    posterior_probabilities = np.prod(class_likelihoods, axis=0) * prior_probabilities

    # Predict the final class with the highest posterior probability.
    final_class_prediction = np.argmax(posterior_probabilities)

    # Calculate the confidence of the final prediction.
    final_confidence = posterior_probabilities[final_class_prediction]

    final_class_prediction = prev_predicted_classes.item(np.argmax(posterior_probabilities))

    # Calculate the weighted average of the current confidence and the final confidence.
    weighted_confidence = (confidence + final_confidence) / 2

    return final_class_prediction, weighted_confidence, massive_object_dictionary


# Define a function to generate random colors for video
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
with open(r'/home/taylordmark/Thesis/Sort/object_dict.txt', 'r') as file:
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
image_folder = Path(r"/home/taylordmark/MOT17/train/MOT17-13-DPM/img1")

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

    # Sort image file paths based on filenames
    image_paths = sorted(image_folder.iterdir(), key=lambda x: int(os.path.splitext(x.name)[0]))

    for image_path in image_paths:
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

        # Naive Bayes should be called in here for each item
        for (xmin, ymin, xmax, ymax, obj_id), (confidence, clas) in zip(track_bbs_ids, confs):
            # If the object has been seen already
            clas = int(clas)
            if obj_id in massive_object_dictionary.keys():
                clas2, confidence2, massive_object_dictionary = naive_bayes(obj_id, confidence, clas, massive_object_dictionary)
                # print(f"{obj_id}: ({clas2}, {confidence2})")
                # Append current confidence and class index
                massive_object_dictionary[obj_id][0] = np.append(massive_object_dictionary[obj_id][0], confidence)
                massive_object_dictionary[obj_id][1] = np.append(massive_object_dictionary[obj_id][1], clas)
                confidence = confidence2
                clas = clas2
            # Otherwise create a new dict index for the id
            else:
                massive_object_dictionary[obj_id] = [np.array(confidence), np.array(clas)]
            
            # Process values
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            clas += 1
            class_name = f"Class {color_mapping[clas]}"

            # Add box data to frame
            object_id_and_class = f"{class_name}: {confidence:.3f}"  # Concatenate Object ID and Confidence
            color = colors[int(clas)] if clas < len(colors) else (0, 0, 0)
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
                'Class': clas
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