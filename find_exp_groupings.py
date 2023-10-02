import random
import matplotlib.pyplot as plt
import numpy as np

# Create an empty dictionary
dictionary = {}

# Populate the dictionary with lists of random values
for key in range(40000):
    lil_list = []
    for _ in range(5):  # Generate 5 values per list
        value = random.uniform(0, 1)
        lil_list.append(value)
    list_sum = sum(lil_list)
    lil_list = [x / list_sum for x in lil_list]
    dictionary[key] = lil_list

# Concatenate all values into a single list
all_values = [value for values in dictionary.values() for value in values]

# Track min and max values
min_value = min(all_values)
max_value = max(all_values)

# Calculate the 25th, 50th (median), and 75th percentiles
percentiles = np.percentile(all_values, [25, 50, 75])

# Print values at each percentile
print(percentiles)

# Create a combined histogram
plt.hist(all_values, bins=10, alpha=0.7, color='b', edgecolor='black')
plt.title(f'Combined Histogram (100x values)\nMin Value: {min_value:.4f}, Max Value: {max_value:.4f}')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Set the x-axis limit to 1
plt.xlim(0, 1)

# Calculate the histogram counts and bin edges
hist, bin_edges = np.histogram(all_values, bins=10)

# Print value counts for each bin with associated bin constraints
for i in range(len(hist)):
    print(f"Bin {i + 1}: {bin_edges[i]:.4f} to {bin_edges[i + 1]:.4f} - Count: {hist[i]}")

plt.show()


print(min_value)
print(max_value)
print(len(all_values))

print()
print(dictionary[0])
print(dictionary[1])
print(dictionary[2])