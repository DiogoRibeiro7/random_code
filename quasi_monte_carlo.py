from scipy.stats import norm, qmc
from typing import Callable
import numpy as np


def quasi_monte_carlo_integration(func: Callable[[np.ndarray], float], bounds: np.ndarray, N: int) -> float:
    """
    Estimate the integral of a function using Quasi-Monte Carlo integration.

    Parameters
    ----------
    func : Callable[[np.ndarray], float]
        The function to integrate.
    bounds : np.ndarray
        The bounds of integration. Should be an array of shape (dim, 2), where dim is the dimension of the input space.
    N : int
        The number of samples to use.

    Returns
    -------
    float
        The estimated value of the integral.
    """

    # Define the dimensionality of the input space
    dim = len(bounds)

    # Create a Sobol sequence generator
    sobol = qmc.Sobol(d=dim)

    # Generate N samples
    samples = sobol.random_base2(m=int(np.log2(N)))

    # Rescale the samples to the bounds of integration
    for i in range(dim):
        samples[:, i] = samples[:, i] * \
            (bounds[i, 1] - bounds[i, 0]) + bounds[i, 0]

    # Evaluate the function at the sampled points
    func_values = func(samples)

    # Compute the average function value
    avg_func_value = np.mean(func_values)

    # Compute the volume of the integration region
    volume = np.prod(bounds[:, 1] - bounds[:, 0])

    # Compute the estimated integral value
    integral_estimate = volume * avg_func_value

    return integral_estimate


# Example usage:
# Define the function to integrate (here, a multivariate normal distribution)
def func(x): return norm.pdf(x)


# Define the bounds of integration
bounds = np.array([[-3, 3]])

# Estimate the integral of the function using Quasi-Monte Carlo integration
print("Estimated integral: ", quasi_monte_carlo_integration(func, bounds, N=10000))


def outlier_detection_quasi_mc(threshold: float, dim: int, N: int) -> float:
    """
    Estimate the proportion of outliers in a data set using Quasi-Monte Carlo sampling.

    Parameters
    ----------
    threshold : float
        The threshold for outlier detection.
    dim : int
        The dimensionality of the data space.
    N : int
        The number of samples to generate.

    Returns
    -------
    float
        The estimated proportion of outliers.
    """

    # Create a Sobol sequence generator
    sobol = qmc.Sobol(d=dim)

    # Generate N samples
    samples = sobol.random_base2(m=int(np.log2(N)))

    # Define the function for detecting outliers
    def outlier_func(x): return x > threshold

    # Apply the function to the samples
    outlier_indicators = np.apply_along_axis(outlier_func, 1, samples)

    # Compute the proportion of outliers
    proportion_outliers = np.mean(outlier_indicators)

    return proportion_outliers


# Example usage:
# Define the threshold for outlier detection
threshold = 0.95

# Define the dimensionality of the data space
dim = 2

# Estimate the proportion of outliers using Quasi-Monte Carlo sampling
print("Estimated proportion of outliers: ",
      outlier_detection_quasi_mc(threshold, dim, N=10000))
