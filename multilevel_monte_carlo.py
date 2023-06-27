import numpy as np
from scipy.stats import norm


def black_scholes_price(S0, K, T, r, sigma):
    """
    The analytic solution of the Black-Scholes equation for a European call option.
    """
    d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def euler_maruyama(S0, K, T, r, sigma, N, M):
    """
    Approximate the option price by the Euler-Maruyama method.
    """
    dt = T / M
    price = np.zeros(N)
    for i in range(N):
        S = S0
        for j in range(M):
            S = S * np.exp((r - sigma**2 / 2) * dt + sigma *
                           np.sqrt(dt) * np.random.normal())
        price[i] = np.maximum(S - K, 0) * np.exp(-r * T)
    return price


def mlmc(S0, K, T, r, sigma, L, N):
    """
    Multilevel Monte Carlo estimator for a European call option.
    """
    sumY = 0.0
    for l in range(L+1):
        M = 2**l
        Y = euler_maruyama(S0, K, T, r, sigma, N, M)
        if l > 0:
            Y_ = euler_maruyama(S0, K, T, r, sigma, N, M//2)
            Y = Y - Y_
        sumY += np.mean(Y)
    return sumY


# Example usage:
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.2
L = 3
N = 1000

print("MLMC estimate: ", mlmc(S0, K, T, r, sigma, L, N))
print("Black-Scholes price: ", black_scholes_price(S0, K, T, r, sigma))


def multilevel_monte_carlo_normal_estimation(data, L, N):
    """
    Estimates the mean and standard deviation of a dataset using multilevel Monte Carlo.
    """
    means = []
    stds = []
    for l in range(L+1):
        M = 2**l
        means_level = []
        stds_level = []
        for i in range(N):
            samples = np.random.choice(data, size=M)
            mean = np.mean(samples)
            std = np.std(samples)
            means_level.append(mean)
            stds_level.append(std)
        means.append(np.mean(means_level))
        stds.append(np.mean(stds_level))
    return np.mean(means), np.mean(stds)


def detect_outliers(data, mean, std, threshold):
    """
    Detects outliers in a dataset, given the mean and standard deviation.
    Outliers are considered as any points that are more than 'threshold' standard deviations away from the mean.
    """
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return outliers


# Create some normally distributed data with a few outliers
true_mean = 0.0
true_std = 1.0
N_data = 1000
data = np.random.normal(loc=true_mean, scale=true_std, size=N_data)
outliers = [5.0, 6.0, -5.0, -6.0]  # these are the outliers
data = np.concatenate((data, outliers))

# Use MLMC to estimate the mean and standard deviation
L = 3
N = 100
estimated_mean, estimated_std = multilevel_monte_carlo_normal_estimation(
    data, L, N)

# Use the estimated mean and standard deviation to detect outliers
threshold = 3.0  # consider as outliers any points that are more than 3 standard deviations away from the mean
detected_outliers = detect_outliers(
    data, estimated_mean, estimated_std, threshold)

print("Estimated mean: ", estimated_mean)
print("Estimated standard deviation: ", estimated_std)
print("Detected outliers: ", detected_outliers)
