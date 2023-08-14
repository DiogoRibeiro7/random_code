import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr
from copulas.bivariate import Clayton
from copulas.univariate import GaussianKDE
from scipy.stats import norm
import numpy as np

# Set random seed for reproducibility
np.random.seed(0)

# Generate synthetic data for temperature and heart rate
sample_size = 1000
temperature = np.random.normal(25, 5, sample_size)
heart_rate = 70 + 0.5 * temperature + np.random.normal(0, 10, sample_size)


temperature_dist = norm.fit(temperature)
heart_rate_dist = norm.fit(heart_rate)


# Create GaussianKDE objects for marginal distributions
temperature_kde = GaussianKDE()
temperature_kde.fit(temperature)

heart_rate_kde = GaussianKDE()
heart_rate_kde.fit(heart_rate)

# Fit Clayton copula to the data
copula = Clayton()
copula.fit(np.column_stack((temperature_kde.cdf(temperature),
                            heart_rate_kde.cdf(heart_rate))))


# Calculate Kendall's tau and Spearman's rho
kendall_tau, _ = kendalltau(temperature, heart_rate)
spearman_rho, _ = spearmanr(temperature, heart_rate)

print("Kendall's Tau:", kendall_tau)
print("Spearman's Rho:", spearman_rho)


# Generate synthetic data using the copula
synthetic_data = copula.sample(sample_size)

# Scatter plot of synthetic data
plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], alpha=0.5)
plt.xlabel('Temperature')
plt.ylabel('Heart Rate')
plt.title('Scatter Plot of Synthetic Data')
plt.show()
