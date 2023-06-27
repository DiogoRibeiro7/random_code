import numpy as np
from typing import Callable

def control_variates(target_func: Callable[[float], float], control_func: Callable[[float], float], control_func_expected: float, sampler: Callable[[], float], N: int) -> float:
    """
    Estimate the expected value of a target function using control variates.

    Parameters
    ----------
    target_func : Callable[[float], float]
        The target function whose expected value is to be estimated.
    control_func : Callable[[float], float]
        The control function whose expected value is known.
    control_func_expected : float
        The known expected value of the control function.
    sampler : Callable[[], float]
        The function to draw samples from the distribution.
    N : int
        The number of samples to draw.

    Returns
    -------
    float
        The estimated expected value of the target function.
    """

    # Draw samples
    samples = np.array([sampler() for _ in range(N)])

    # Evaluate the target and control functions at the sampled points
    target_values = target_func(samples)
    control_values = control_func(samples)

    # Compute the sample covariance and variance
    covariance = np.cov(target_values, control_values)[0, 1]
    variance = np.var(control_values)

    # Compute the optimal coefficient
    b_star = -covariance / variance

    # Compute the control variates estimator
    estimator = np.mean(target_values + b_star * (control_values - control_func_expected))

    return estimator

# Example usage:
np.random.seed(0)  # for reproducibility

# Define the target function (here, the square function)
target_func = lambda x: x**2

# Define the control function (here, the identity function)
control_func = lambda x: x

# The expected value of the control function for a standard normal distribution is 0
control_func_expected = 0

# Define the sampler function to draw samples from a standard normal distribution
sampler = lambda: np.random.normal()

# Estimate the expected value of the target function using control variates
print("Estimated expected value: ", control_variates(target_func, control_func, control_func_expected, sampler, N=10000))
