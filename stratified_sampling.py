import numpy as np
from scipy.stats import norm

# suppose this is your data
heart_beats = np.random.normal(loc=70, scale=10, size=10000)  # normally distributed for simplicity

# stratify into 10 strata
strata_bounds = np.linspace(np.min(heart_beats), np.max(heart_beats), 11)

# these will hold the stratified samples
stratified_samples = []

for i in range(10):
    stratum_heart_beats = heart_beats[(heart_beats >= strata_bounds[i]) & (heart_beats < strata_bounds[i+1])]
    
    # calculate the weights for this stratum
    weights = norm.pdf(stratum_heart_beats, loc=70, scale=10)
    weights /= np.sum(weights)
    
    # sample from this stratum with replacement
    stratified_sample = np.random.choice(stratum_heart_beats, size=len(stratum_heart_beats), p=weights)
    
    stratified_samples.extend(stratified_sample)

# compute the 5% percentile of the stratified samples
percentile_5 = np.percentile(stratified_samples, 5)
print(f"5% Percentile of Heart Beats per Minute: {percentile_5}")




import numpy as np
from scipy.stats import norm

# Suppose this is your data
heart_beats = np.random.normal(loc=70, scale=10, size=10000)  # normally distributed for simplicity

# Initial guess for the mean and standard deviation
mu, sigma = np.mean(heart_beats), np.std(heart_beats)

# Number of iterations for adaptation
num_iterations = 50

# Number of samples per iteration
num_samples = 1000

# Our aim is to model the distribution of heart beats per minute
# So, we use the data itself to compute the weights
f = lambda x: np.sum(norm(loc=heart_beats, scale=1.0).pdf(x))

for i in range(num_iterations):
    # Draw samples from the proposal distribution
    samples = np.random.normal(loc=mu, scale=sigma, size=num_samples)
    
    # Compute weights for the samples
    weights = np.array([f(x) for x in samples]) / norm(loc=mu, scale=sigma).pdf(samples)
    
    # Normalize the weights
    weights /= np.sum(weights)
    
    # Update the mean and standard deviation based on the weights
    mu = np.sum(weights * samples)
    sigma = np.sqrt(np.sum(weights * (samples - mu)**2))

# Now, draw samples from the final proposal distribution and compute the 5th percentile
samples = np.random.normal(loc=mu, scale=sigma, size=10000)
percentile_5 = np.percentile(samples, 5)

print(f"5% Percentile of Heart Beats per Minute using Adaptive Importance Sampling: {percentile_5}")
