import numpy as np


# This is the part I'm worried about
# # It may take a while to figure out how to do this
# Alternatively, could do it separately for a sample of frames in a few videos

# Define the original probability distributions (from training data)
distributions = {
    "person": np.array([.7, .1999, .1001, .0, .0]),
    "bicycle": np.array([0.2, 0.6, 0.05, 0.3, 0.05]),
    "car": np.array([0.1, 0.3, 0.6, 0.1, 0.0]),
    "motorcycle": np.array([0.2, 0.2, 0.0, 0.55, 0.05]),
    "traffic light": np.array([0.1, 0.1, 0.0, 0.1, 0.7])}


# [person, bicycle,  car,  motorcycle,  traffic light]



# Define the current probability distribution

# Person-esque prediction
new_distribution = np.array([0.6, 0.2, 0.2, 0.0, 0.0])

# Traffic light-esque prediction
# new_distribution = np.array([.0, .1, .1, .3, .5])

# Bicycle-esque prediction
# new_distribution = np.array([.1, .6, .0, .2, .1])




# Define the population percentages for each class
population_counts = {
    "person": 501997,  
    "bicycle": 24892,  
    "car": 24300,     
    "motorcycle": 11261,  
    "traffic light": 1778}

# Apply a log transformation to the population percentages
# This will only occur in the first iteration, after which the number of detected
# objects will be added at each iteration
population_percentages = {key: np.log(value) for key, value in population_counts.items()}

# Step 1: Calculate Attention Scores - class dists (dot) new dist
attention_scores = {}
for key, distribution in distributions.items():
    attention_score = np.dot(new_distribution, distribution)
    attention_scores[key] = attention_score

# Step 2: Calculate weighted scores for each class dist
# given P(A)/P(B) (percent of total population)
attention_weights = {}
total_log_population_percentage = sum(population_percentages.values())
for key, score in attention_scores.items():
    # Weight attention score by transformed population percentage
    weighted_score = score * (population_percentages[key] / total_log_population_percentage)
    attention_weights[key] = weighted_score

# Step 3: Calculate Weighted Average (attention scores * attention weights)
weighted_sums = {}
for key in distributions.keys():
    weighted_sum = np.sum(attention_weights[key])
    weighted_sums[key] = weighted_sum

# Normalize the similarities so that their sum equals 1
sum_similarity = sum(weighted_sums.values())
normalized_similarities = {key: value / sum_similarity for key, value in weighted_sums.items()}

# Print normalized similarities
for key, value in normalized_similarities.items():
    print(f"Normalized Similarity with {key}: {value:.4f}")

# Find the distribution with the highest normalized similarity
max_normalized_similarity_key = max(normalized_similarities, key=normalized_similarities.get)
max_normalized_similarity_value = normalized_similarities[max_normalized_similarity_key]

# Print the highest normalized similarity and the corresponding distribution
print(f"\nHighest Normalized Similarity: {max_normalized_similarity_value:.4f} with {max_normalized_similarity_key}")
