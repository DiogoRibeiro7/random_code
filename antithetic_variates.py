import numpy as np

def estimate_pi_antithetic(n_samples: int) -> float:
    """
    Estimate the value of Pi using Monte Carlo simulation with Antithetic Variates.
    
    Parameters
    ----------
    n_samples : int
        The number of random samples to generate.

    Returns
    -------
    float
        The estimated value of Pi.

    Raises
    ------
    ValueError
        If `n_samples` is not a positive even integer.
    """

    # Check that `n_samples` is a positive even integer
    if not isinstance(n_samples, int) or n_samples <= 0 or n_samples % 2 != 0:
        raise ValueError("`n_samples` must be a positive even integer.")
    
    # Initialize count of points inside the circle
    count_inside_circle = 0

    for _ in range(n_samples // 2):
        # Generate a pair of random points in [0, 1] x [0, 1]
        point1 = np.random.rand(2)
        point2 = 1 - point1  # Antithetic point

        # Check if the points are inside the circle
        if np.sum(point1**2) < 1:
            count_inside_circle += 1
        if np.sum(point2**2) < 1:
            count_inside_circle += 1

    # Estimate Pi
    pi_estimate = 4 * count_inside_circle / n_samples

    return pi_estimate

# Test the function
n_samples = 1000000
pi_estimate = estimate_pi_antithetic(n_samples)
print("Estimated Pi:", pi_estimate)



import numpy as np
from scipy.stats import norm

def european_call_option_price(S, K, r, sigma, T):
    """
    Compute the price of a European call option using the Black-Scholes formula.

    Parameters
    ----------
    S : float
        The current price of the underlying asset.
    K : float
        The strike price of the option.
    r : float
        The risk-free interest rate.
    sigma : float
        The volatility of the underlying asset.
    T : float
        The time to maturity of the option.

    Returns
    -------
    float
        The price of the European call option.
    """
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def antithetic_variates_european_call_option_price(S, K, r, sigma, T, n_samples):
    """
    Estimate the price of a European call option using Monte Carlo simulation with Antithetic Variates.

    Parameters
    ----------
    S : float
        The current price of the underlying asset.
    K : float
        The strike price of the option.
    r : float
        The risk-free interest rate.
    sigma : float
        The volatility of the underlying asset.
    T : float
        The time to maturity of the option.
    n_samples : int
        The number of random samples to generate.

    Returns
    -------
    float
        The estimated price of the European call option.
    """
    # Check that `n_samples` is a positive even integer
    if not isinstance(n_samples, int) or n_samples <= 0 or n_samples % 2 != 0:
        raise ValueError("`n_samples` must be a positive even integer.")
    
    # Generate random samples
    samples1 = np.random.normal(size=n_samples // 2)
    samples2 = -samples1  # Antithetic variates

    # Compute call option prices for each sample
    call_prices1 = european_call_option_price(S, K, r, sigma, T) * np.exp(r * T)  # Discounted call prices
    call_prices2 = european_call_option_price(S, K, r, sigma, T) * np.exp(r * T)  # Discounted call prices

    # Estimate the option price using the mean of the call prices
    option_price = 0.5 * (np.mean(call_prices1) + np.mean(call_prices2))

    return option_price

# Test the function
S = 100.0  # Current price of the underlying asset
K = 100.0  # Strike price of the option
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility of the underlying asset
T = 1.0  # Time to maturity of the option
n_samples = 1000000  # Number of samples

option_price = antithetic_variates_european_call_option_price(S, K, r, sigma, T, n_samples)
print("Estimated Option Price:", option_price)


import numpy as np
from scipy.stats import norm

def detect_outliers_antithetic_variates(data):
    """
    Detect outliers in a dataset using the Z-score method with Antithetic Variates.

    Parameters
    ----------
    data : array-like
        The input data.

    Returns
    -------
    list
        A list of indices corresponding to the detected outliers.
    """
    # Convert the data to a NumPy array
    data = np.asarray(data)

    # Compute the mean and standard deviation of the data
    mean = np.mean(data)
    std = np.std(data)

    # Generate antithetic variates
    antithetic_data = -data

    # Compute Z-scores for the original and antithetic data
    z_scores = (data - mean) / std
    antithetic_z_scores = (antithetic_data - mean) / std

    # Concatenate the Z-scores
    all_z_scores = np.concatenate((z_scores, antithetic_z_scores))

    # Set a threshold for outlier detection (e.g., Z-score > 3)
    threshold = 3.0

    # Detect outliers based on the Z-scores
    outlier_indices = np.where(np.abs(all_z_scores) > threshold)[0]

    return outlier_indices.tolist()

# Generate some normally distributed data with outliers
data = np.concatenate((np.random.normal(loc=0, scale=1, size=1000),
                       np.random.normal(loc=10, scale=1, size=10)))  # Adding outliers

# Detect outliers using antithetic variates and the Z-score method
outlier_indices = detect_outliers_antithetic_variates(data)

print("Detected outlier indices:", outlier_indices)

