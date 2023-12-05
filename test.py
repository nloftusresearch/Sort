import numpy as np
from scipy.stats import beta

# Your list of numbers
data = [.1,.2,.4]
new_number = .7
# Fit a beta distribution to the data
alpha, beta_params, loc, scale = beta.fit(data, floc=0, fscale=1)

# Parameters of the fitted beta distribution
print(f"Fitted Parameters: alpha = {alpha}, beta = {beta_params}, loc = {loc}, scale = {scale}")

# Calculate the PDF for the new number
pdf = beta.pdf(new_number, alpha, beta_params, loc, scale)
print(f"PDF for {new_number} = {pdf}")
