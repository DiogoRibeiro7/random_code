from typing import Callable
from typing import List, Tuple
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


def stratified_sampling(alpha: float, beta: float, N: int, num_strata: int) -> float:
    """
    Estimate the mean of a Beta distribution using stratified sampling.

    Parameters
    ----------
    alpha : float
        The first shape parameter of the Beta distribution.
    beta : float
        The second shape parameter of the Beta distribution.
    N : int
        The total number of samples to draw.
    num_strata : int
        The number of strata to use in the stratified sampling.

    Returns
    -------
    float
        The estimated mean of the distribution.
    """

    # Define the target distribution
    target = stats.beta(a=alpha, b=beta)

    # Number of samples per stratum
    N_stratum = N // num_strata

    # Initialize the sum of sample means
    sum_sample_means = 0.0

    # Perform stratified sampling
    for i in range(num_strata):
        # Define the stratum limits
        a_stratum = i / num_strata
        b_stratum = (i + 1) / num_strata

        # Draw samples uniformly within the stratum
        samples_stratum = np.random.uniform(
            low=a_stratum, high=b_stratum, size=N_stratum)

        # Compute the sample mean within the stratum
        sample_mean_stratum = np.mean(target.pdf(samples_stratum))

        # Add the sample mean to the sum of sample means
        sum_sample_means += sample_mean_stratum

    # Divide by the number of strata to get the estimate of the mean
    mean_estimate = sum_sample_means / num_strata

    return mean_estimate


# Test the function with an example
print("Estimated mean: ", stratified_sampling(
    alpha=2, beta=5, N=10000, num_strata=10))


def recursive_stratified_sampling(func: Callable[[float], float], a: float, b: float, depth: int) -> float:
    """
    Estimate the integral of a function using recursive stratified sampling.

    Parameters
    ----------
    func : Callable[[float], float]
        The function to integrate.
    a : float
        The lower limit of integration.
    b : float
        The upper limit of integration.
    depth : int
        The maximum depth of recursion.

    Returns
    -------
    float
        The estimated integral of the function.
    """

    if depth == 0:
        # Base case: Estimate the integral using the midpoint rule
        x = (a + b) / 2.0
        return (b - a) * func(x)
    else:
        # Recursive case: Split the interval and estimate the integral in each half
        c = (a + b) / 2.0
        return recursive_stratified_sampling(func, a, c, depth - 1) + recursive_stratified_sampling(func, c, b, depth - 1)


# Test the function with an example
print("Estimated integral: ", recursive_stratified_sampling(
    func=np.sin, a=0, b=np.pi, depth=10))


def monte_carlo_importance_sampling(target_dist: Callable[[float], float], proposal_dist: Callable[[float], float], proposal_sampler: Callable[[], float], N: int) -> float:
    """
    Estimate the expectation of a target distribution using Monte Carlo integration with importance sampling.

    Parameters
    ----------
    target_dist : Callable[[float], float]
        The target distribution.
    proposal_dist : Callable[[float], float]
        The proposal distribution.
    proposal_sampler : Callable[[], float]
        The function to draw samples from the proposal distribution.
    N : int
        The number of samples to draw.

    Returns
    -------
    float
        The estimated expectation of the target distribution.
    """

    # Draw samples from the proposal distribution
    samples = np.array([proposal_sampler() for _ in range(N)])

    # Compute the weights for each sample
    weights = target_dist(samples) / proposal_dist(samples)

    # Compute the weighted average of the samples
    expectation_estimate = np.sum(samples * weights) / np.sum(weights)

    return expectation_estimate


# Define the target distribution as a standard normal distribution
target_dist = stats.norm(loc=0, scale=1).pdf

# Define the proposal distribution as a wider normal distribution
proposal_dist = stats.norm(loc=0, scale=2).pdf

# Define the function to draw samples from the proposal distribution
proposal_sampler = stats.norm(loc=0, scale=2).rvs

# Estimate the expectation using Monte Carlo integration with importance sampling
print("Estimated expectation: ", monte_carlo_importance_sampling(
    target_dist, proposal_dist, proposal_sampler, N=10000))
