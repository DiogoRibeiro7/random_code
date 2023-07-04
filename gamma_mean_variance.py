import numpy as np
import matplotlib.pyplot as plt

def sample_gamma_from_mean_variance(mean, variance, min_value, max_value, n_samples):
    # Calculate shape and scale from mean and variance
    shape = (mean ** 2) / variance
    scale = variance / mean

    samples = []
    while len(samples) < n_samples:
        sample = np.random.gamma(shape, scale)
        if min_value <= sample <= max_value:
            samples.append(sample)
    return samples

# Using the function
mean = 80
variance = 30  # Adjust this value to experiment with the spread
n_samples = 10000

gamma_samples = sample_gamma_from_mean_variance(mean, variance, 50, 200, n_samples)

# Plotting
plt.hist(gamma_samples, bins=30, density=True, alpha=0.6, color='g')
plt.title("Gamma Distribution with mean {} and variance {}".format(mean, variance))
plt.show()
