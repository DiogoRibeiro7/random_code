import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def metropolis_hastings(target_density, proposal_sampler, initial_value, n_samples):
    samples = [initial_value]
    current_value = initial_value

    while len(samples) < n_samples:
        proposal = proposal_sampler(current_value)

        # Calculate the acceptance ratio
        ratio = target_density(proposal) / target_density(current_value)

        # Accept or reject the proposal
        if np.random.uniform(0, 1) < min(1, ratio):
            current_value = proposal

        samples.append(current_value)

    return np.array(samples)

# Target density function: mixture of two Gaussians
def target_density(x):
    return 0.5 * norm.pdf(x, -2, 1) + 0.5 * norm.pdf(x, 2, 1)

# Proposal sampler function: uniform distribution
def proposal_sampler(current):
    return np.random.uniform(current - 0.5, current + 0.5)

# Generate samples using the Metropolis-Hastings algorithm
samples = metropolis_hastings(target_density, proposal_sampler, initial_value=0.0, n_samples=50000)

# Plot the histogram of the samples
plt.hist(samples, bins=50, density=True, alpha=0.6)

# Plot the true mixture of two Gaussians
x = np.linspace(-6, 6, 1000)
plt.plot(x, target_density(x), 'r', lw=2)

plt.show()
