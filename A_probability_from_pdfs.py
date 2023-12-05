import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import beta

def read_data(file_path):
    # Read data from the specified text file
    object_type_distributions = []
    with open(file_path, 'r') as file:
        for line in file:
            # Parse the line to extract relevant information
            frame_data = eval(line.strip())
            
            # Extract class probabilities and convert them to a numpy array
            object_features = np.array(frame_data["correct_prob"])
            
            # Append the numpy array to the list
            object_type_distributions.append(object_features)
    return object_type_distributions

def generate_plot(num_classes, object_type_distributions):
    # Define the number of rows (i) and columns (j) for the panel
    i, j = num_classes, num_classes
    # Create subplots for the i x j panel
    fig, axes = plt.subplots(i, j, figsize=(15, 10))
    # Loop through each combination of columns and create histograms
    for row in range(i):
        for col in range(j):
            # Plot the histogram
            axes[row, col].hist(object_type_distributions[row][:, col], bins=10)  # Adjust the number of bins as needed
            axes[row, col].set_title(f'Column {col + 1} Histogram for Object Type {row + 1}')
            axes[row, col].set_xlabel('Values')
            axes[row, col].set_ylabel('Frequency')

            # Set the x-axis limits to match the maximum value
            axes[row, col].set_xlim(0, max_value)


    # Display the histograms
    plt.tight_layout()
    plt.show()

def get_prob_from_dist(x, parameters):
  """Returns the probability of a number being from a right skewed distribution using all three parameters in a list: shape, scale, and loc, respectively.

  Args:
    x: The number whose probability you want to calculate.
    parameters: A list of the values for a, b, lc, scale respectively

  Returns:
    The probability of the number being from the right skewed distribution.
  """

  # Calculate the probability of the number being from the gamma distribution.
  probability = beta.pdf(x, parameters[0], parameters[1], loc=parameters[2], scale=parameters[3])

  return probability

def get_class_probabilities(population_counts):
    total_population = sum(population_counts.values())
    class_probabilities = {key: value / total_population for key, value in population_counts.items()}
    return class_probabilities

# Define the labels for each class
class_dictionary = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "11",
    12: "12",
    13: "13"
}


# Define the population percentages for each class
population_counts = {
    0: 501997,
    1: 24892,
    2: 24300,
    3: 11261,
    4: 1778,
    5: 12345,
    6: 5432,
    7: 9876,
    8: 5678,
    9: 9876,
    10: 3456,
    11: 2345,
    12: 7890,
    13: 4567
}

# Log transform initial population values
population_counts = {key: np.log(value) for key, value in population_counts.items()}

# Calculate probabilities based on the log-transformed population percentages
class_probabilities = get_class_probabilities(population_counts)


#Reading data from .txt file

# Specify the path to your data file
data_file_path = '/remote_home/Thesis/Sort/frames_output.txt'

# Read data from the file
object_type_distributions = read_data(data_file_path)


# Iterate through each frame and make predictions
for frame_data in object_type_distributions:
    distribution_parameters = []
    for row in range(len(class_dictionary)):
        row_parameters = []
        for col in range(len(class_dictionary)):
            data = frame_data[:, col]

            # Calculate distribution statistics
            a, b, loc, scale = beta.fit(data)
            # Store parameters in a dictionary
            row_parameters.append([a, b, loc, scale])
        distribution_parameters.append(row_parameters)

    pdfs = []
    for row in distribution_parameters:
        p = 1
        for index, (a, b, loc, scale) in enumerate(row):
            probability = beta.pdf(new_distribution[index], a, b, loc=loc, scale=scale)
            # Add a little so none multiplied by 0
            probability += 0.001
            p *= probability
        pdfs.append(p)

    # Multiply by population probabilities
    pdfs = [pdfs[i] * class_probabilities[i] for i in range(len(pdfs))]

    prediction = pdfs.index(max(pdfs))
    print(f"\nClass prediction for this frame: {class_dictionary[prediction]}")
