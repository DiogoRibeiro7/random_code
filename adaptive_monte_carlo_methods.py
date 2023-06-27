import numpy as np
from scipy.stats import norm


def rare_event_simulation_adaptive_monte_carlo(target_mu, target_sigma, threshold, N):
    """
    Perform rare event simulation using Adaptive Monte Carlo with Importance Sampling.

    Parameters
    ----------
    target_mu : float
        Mean of the target distribution.
    target_sigma : float
        Standard deviation of the target distribution.
    threshold : float
        Threshold value for the rare event.
    N : int
        Number of samples to generate.

    Returns
    -------
    float
        Estimate of the probability of the rare event.
    """
    # Define the target and proposal distributions
    target = norm(loc=target_mu, scale=target_sigma)
    # Adjusted proposal distribution
    proposal = norm(loc=target_mu, scale=target_sigma/2)

    # Initialize variables for weighted sum and normalization
    weighted_sum = 0.0
    normalization = 0.0

    for _ in range(N):
        # Generate a sample from the proposal distribution
        sample = proposal.rvs()

        # Compute the importance weight
        weight = target.pdf(sample) / proposal.pdf(sample)

        # Check if the sample is in the rare event region
        if sample > threshold:
            weighted_sum += weight
            normalization += 1.0

    # Estimate the probability of the rare event
    prob_estimate = weighted_sum / \
        (normalization + 1e-10)  # Avoid division by zero

    return prob_estimate


# Example usage
target_mu = 5.0
target_sigma = 1.0
threshold = 10.0
N = 100000

prob_estimate = rare_event_simulation_adaptive_monte_carlo(
    target_mu, target_sigma, threshold, N)
print("Probability estimate:", prob_estimate)


def outlier_detection_adaptive_monte_carlo(data, threshold, N):
    """
    Perform outlier detection using Adaptive Monte Carlo with Importance Sampling.

    Parameters
    ----------
    data : array-like
        The input data.
    threshold : float
        Threshold value for defining outliers.
    N : int
        Number of samples to generate.

    Returns
    -------
    list
        A list of indices corresponding to the detected outliers.
    """
    # Convert the data to a NumPy array
    data = np.asarray(data)

    # Compute the mean and standard deviation of the data
    data_mu = np.mean(data)
    data_sigma = np.std(data)

    # Define the target and proposal distributions
    target = norm(loc=data_mu, scale=data_sigma)
    # Adjusted proposal distribution
    proposal = norm(loc=data_mu, scale=data_sigma/2)

    # Initialize a list to store outlier indices
    outlier_indices = []

    for _ in range(N):
        # Generate a sample from the proposal distribution
        sample = proposal.rvs()

        # Compute the importance weight
        weight = target.pdf(sample) / proposal.pdf(sample)

        # Check if the sample is an outlier based on the threshold
        if sample > threshold:
            # Sample is considered an outlier
            # Get the index of the first occurrence
            outlier_indices.append(np.argmax(data > sample))

    return outlier_indices


# Example usage
data = np.concatenate((np.random.normal(loc=0, scale=1, size=1000),
                       np.random.normal(loc=10, scale=1, size=10)))  # Adding outliers
threshold = 3.0
N = 10000

outlier_indices = outlier_detection_adaptive_monte_carlo(data, threshold, N)
print("Detected outlier indices:", outlier_indices)
