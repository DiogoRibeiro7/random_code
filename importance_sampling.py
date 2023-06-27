from typing import Tuple
import numpy as np
import scipy.stats as stats


def estimate_expectation(target_mu: float, target_sigma: float, proposal_mu: float, proposal_sigma: float, N: int) -> float:
    """
    Estimate the expectation of a standard normal random variable outside the range (-1, 1) using importance sampling.

    Parameters
    ----------
    target_mu : float
        The mean of the target distribution.
    target_sigma : float
        The standard deviation of the target distribution.
    proposal_mu : float
        The mean of the proposal distribution.
    proposal_sigma : float
        The standard deviation of the proposal distribution.
    N : int
        The number of samples to draw from the proposal distribution.

    Returns
    -------
    float
        The estimated expectation of the random variable.
    """

    # Define the target and proposal distributions
    target = stats.norm(loc=target_mu, scale=target_sigma)
    proposal = stats.norm(loc=proposal_mu, scale=proposal_sigma)

    # Draw N samples from the proposal distribution
    samples_proposal = proposal.rvs(size=N)

    # Compute the weights for each sample
    # The weights are the ratio of the pdf of the target distribution to the pdf of the proposal distribution
    weights = target.pdf(samples_proposal) / proposal.pdf(samples_proposal)

    # Define the indicator function which is 1 if the absolute value of the sample is greater than 1 and 0 otherwise
    ind_func = np.where(np.abs(samples_proposal) > 1, 1, 0)

    # Compute the weighted sum of the samples where the weights are modified by the indicator function
    weighted_sum = np.sum(samples_proposal * weights * ind_func)

    # Divide the weighted sum by the sum of the weights times the indicator function to get the estimate of the expectation
    expectation_estimate = weighted_sum / np.sum(weights * ind_func)

    return expectation_estimate


# Test the function with an example
print("Estimated expectation: ", estimate_expectation(
    target_mu=0, target_sigma=1, proposal_mu=0, proposal_sigma=2, N=10000))


def estimate_probability(target_mu: float, target_sigma: float, proposal_mu: float, proposal_sigma: float, threshold: float, N: int) -> float:
    """
    Estimate the probability of a rare event (when a random variable exceeds a threshold) using importance sampling.

    Parameters
    ----------
    target_mu : float
        The mean of the target distribution.
    target_sigma : float
        The standard deviation of the target distribution.
    proposal_mu : float
        The mean of the proposal distribution.
    proposal_sigma : float
        The standard deviation of the proposal distribution.
    threshold : float
        The threshold that defines the rare event (we're estimating the probability that a random variable from the target distribution is greater than this value).
    N : int
        The number of samples to draw from the proposal distribution.

    Returns
    -------
    float
        The estimated probability of the rare event.
    """

    # Define the target and proposal distributions
    target = stats.norm(loc=target_mu, scale=target_sigma)
    proposal = stats.norm(loc=proposal_mu, scale=proposal_sigma)

    # Draw N samples from the proposal distribution
    samples_proposal = proposal.rvs(size=N)

    # Compute the weights for each sample
    # The weights are the ratio of the pdf of the target distribution to the pdf of the proposal distribution
    weights = target.pdf(samples_proposal) / proposal.pdf(samples_proposal)

    # Define the indicator function which is 1 if the sample is greater than the threshold and 0 otherwise
    ind_func = np.where(samples_proposal > threshold, 1, 0)

    # Compute the weighted sum of the indicator function values
    weighted_sum = np.sum(ind_func * weights)

    # Divide the weighted sum by the number of samples to get the estimate of the probability
    prob_estimate = weighted_sum / N

    return prob_estimate


# Test the function with an example
print("Estimated probability: ", estimate_probability(target_mu=0,
      target_sigma=1, proposal_mu=0, proposal_sigma=1, threshold=3, N=10000))
