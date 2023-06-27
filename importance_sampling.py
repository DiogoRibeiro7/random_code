import numpy as np
import scipy.stats as stats

def estimate_probability(target_mu, target_sigma, proposal_mu, proposal_sigma, threshold, N):
    # Define the target and proposal distributions
    target = stats.norm(loc=target_mu, scale=target_sigma)
    proposal = stats.norm(loc=proposal_mu, scale=proposal_sigma)

    # Draw samples from the proposal distribution
    samples_proposal = proposal.rvs(size=N)

    # Compute weights
    weights = target.pdf(samples_proposal) / proposal.pdf(samples_proposal)

    # Compute the indicator function (1 if sample > threshold, 0 otherwise)
    ind_func = np.where(samples_proposal > threshold, 1, 0)

    # Compute the weighted sum of the indicator function
    weighted_sum = np.sum(ind_func * weights)

    # Divide by the number of samples to get the estimate of the probability
    prob_estimate = weighted_sum / N

    return prob_estimate

# Test the function with an example
print("Estimated probability: ", estimate_probability(target_mu=0, target_sigma=1, proposal_mu=0, proposal_sigma=1, threshold=3, N=10000))
