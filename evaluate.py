from asyncio.proactor_events import _ProactorBasePipeTransport
from traceback import StackSummary
from motmetrics import metrics, utils
import csv
import numpy as np

def get_ground_truth(ground_truth_file):
    # Create a dictionary to store ground truth data
    gt = {}

    with open(ground_truth_file, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            frame_number = int(parts[0])
            object_id = int(parts[1])
            x = int(parts[2])
            y = int(parts[3])
            width = int(parts[4])
            height = int(parts[5])
            confidence = float(parts[6])
            class_id = int(parts[7])
            

            if frame_number not in gt:
                gt[frame_number] = []

            gt[frame_number].append({
                'FrameNumber': frame_number,
                'ObjectID': object_id,
                'X': x,
                'Y': y,
                'Width': width,
                'Height': height,
                'Confidence': confidence,
                'Class': class_id  # Assuming class information is in the 8th column
            })
    return gt

def get_tracking_data(tracking_file):
    # Define a dictionary to store tracking results
    tracking_results = {}

    # Open the CSV file and read tracking results
    with open(tracking_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            frame_number = int(row['FrameNumber'])
            object_id = int(row['ObjectID'])
            x = int(row['X'])
            y = int(row['Y'])
            width = int(row['Width'])
            height = int(row['Height'])
            confidence = float(row['Confidence'])
            
            # Handle non-integer 'Class' values
            try:
                class_id = int(row['Class'])
            except ValueError:
                class_id = -1  # Set a default value, or you can handle this differently
            
            if frame_number not in tracking_results:
                tracking_results[frame_number] = []

            tracking_results[frame_number].append({
                'FrameNumber': frame_number,
                'ObjectID': object_id,
                'X': x,
                'Y': y,
                'Width': width,
                'Height': height,
                'Confidence': confidence,
                'Class': class_id
            })
    return tracking_results

def fill_accumulator(ground_truth, tracked_objects, accumulator):
    # Update accumulator with ground truth and tracking results
    for frame_number in ground_truth.keys():
        gt_frame = ground_truth[frame_number]
        track_frame = tracked_objects.get(frame_number, [])

        # Calculate the Euclidean distance manually
        dists = np.zeros((len(gt_frame), len(track_frame)))
        for i, gt_obj in enumerate(gt_frame):
            for j, track_obj in enumerate(track_frame):
                gt_x, gt_y = gt_obj['X']+gt_obj['Width']*.5, gt_obj['Y']+gt_obj['Height']*.5
                track_x, track_y = track_obj['X']+track_obj['Width']*.5, track_obj['Y']+track_obj['Height']*.5
                dist = np.sqrt((gt_x - track_x) ** 2 + (gt_y - track_y) ** 2)
                dists[i][j] = dist
    
        gt_ids = []
        track_ids = []
    
        for item in gt_frame:
            gt_id = item.get('ObjectID')
            gt_ids.append(gt_id)
        for item in track_frame:
            track_id = item.get('ObjectID')
            track_ids.append(track_id)

        acc.update(gt_ids, track_ids, dists)
    return accumulator


# Read ground truth data from the 'gt.txt' file
gt_file_path = r"D:\Coding\Thesis\MOT17\train\MOT17-13-DPM\gt\gt.txt"

# Specify the path to the tracking results CSV file
# tracking_results_file = 'just_detect_results.csv'
tracking_results_file = 'detect_and_sort_results.csv'

ground_truth = get_ground_truth(gt_file_path)
tracked_objects = get_tracking_data(tracking_results_file)

# Create an accumulator to compute tracking metrics
acc = metrics.MOTAccumulator(auto_id=True)

acc = fill_accumulator(ground_truth, tracked_objects, acc)

# Calculate tracking metrics
mh = metrics.create()
summary = mh.compute(acc, metrics=['num_frames', 'num_false_positives', 'num_misses', 'num_switches', 'num_objects', 'mota', 'motp'], name='acc')

# Print the summary of tracking metrics
for key, value in summary.items():
    print(key)
    print(f"{value[0]}\n")