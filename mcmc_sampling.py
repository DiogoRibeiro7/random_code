import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from copy import copy
from typing import List, Optional

class Sampler:
    """
    Sampler class to perform Markov chain Monte Carlo (MCMC) simulation.

    Example usage:
    ```
    data = np.random.randn(20)
    s = Sampler(data=data)
    posterior = s.sampler(samples=5000, mu_init=1, proposal_width=0.5, plot=False)
    print(posterior)
    ```
    """
    def __init__(self, data: np.ndarray, mu_prior_mu: float = 0, mu_prior_sd: float = 1.):
        """
        Initialize the sampler with the data and prior parameters.

        Args:
            data: numpy array of data points.
            mu_prior_mu: mean of the prior distribution.
            mu_prior_sd: standard deviation of the prior distribution.
        """
        self.data = data
        self.mu_prior_mu = mu_prior_mu
        self.mu_prior_sd = mu_prior_sd

    def sampler(self, samples: int = 4, mu_init: float = .5, proposal_width: float = .5, plot: bool = False) -> np.ndarray:
        """
        Perform the MCMC simulation.

        Args:
            samples: number of samples to draw.
            mu_init: initial mu value.
            proposal_width: width of the proposal distribution.
            plot: whether to plot the proposal at each step.

        Returns:
            numpy array of the posterior samples.
        """
        mu_current = mu_init
        posterior = [mu_current]
        for i in range(samples):
            mu_proposal = norm(mu_current, proposal_width).rvs()
            likelihood_current = norm(mu_current, 1).pdf(self.data).prod()
            likelihood_proposal = norm(mu_proposal, 1).pdf(self.data).prod()

            prior_current = norm(self.mu_prior_mu, self.mu_prior_sd).pdf(mu_current)
            prior_proposal = norm(self.mu_prior_mu, self.mu_prior_sd).pdf(mu_proposal)

            p_current = likelihood_current * prior_current
            p_proposal = likelihood_proposal * prior_proposal

            p_accept = p_proposal / p_current

            accept = np.random.rand() < p_accept

            if plot:
                self.plot_proposal(mu_current, mu_proposal, accept, posterior, i)

            if accept:
                mu_current = mu_proposal

            posterior.append(mu_current)

        return np.array(posterior)

def plot_proposal(self, mu_current: float, mu_proposal: float, accepted: bool, trace: List[float], i: int) -> None:
    """
    Plot the proposal at the current step.

    Args:
        mu_current: current mu value.
        mu_proposal: proposed mu value.
        accepted: whether the proposal was accepted.
        trace: trace of the mu values.
        i: current step.
    """
    trace = copy(trace)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(16, 4))
    fig.suptitle('Iteration %i' % (i + 1))
    x = np.linspace(-3, 3, 5000)
    color = 'g' if accepted else 'r'

    # Plot prior
    prior_current = norm(self.mu_prior_mu, self.mu_prior_sd).pdf(mu_current)
    prior_proposal = norm(self.mu_prior_mu, self.mu_prior_sd).pdf(mu_proposal)
    prior = norm(self.mu_prior_mu, self.mu_prior_sd).pdf(x)
    ax1.plot(x, prior)
    ax1.plot([mu_current] * 2, [0, prior_current], marker='o', color='b')
    ax1.plot([mu_proposal] * 2, [0, prior_proposal], marker='o', color=color)
    ax1.annotate("", xy=(mu_proposal, 0.2), xytext=(mu_current, 0.2),
                 arrowprops=dict(arrowstyle="->", lw=2.))
    ax1.set(ylabel='Probability Density', title='current: prior(mu=%.2f) = %.2f\nproposal: prior(mu=%.2f) = %.2f' % (mu_current, prior_current, mu_proposal, prior_proposal))

    # Likelihood
    likelihood_current = norm(mu_current, 1).pdf(self.data).prod()
    likelihood_proposal = norm(mu_proposal, 1).pdf(self.data).prod()
    y = norm(loc=mu_proposal, scale=1).pdf(x)
    sns.distplot(self.data, kde=False, norm_hist=True, ax=ax2)
    ax2.plot(x, y, color=color)
    ax2.axvline(mu_current, color='b', linestyle='--', label='mu_current')
    ax2.axvline(mu_proposal, color=color, linestyle='--', label='mu_proposal')
    ax2.annotate("", xy=(mu_proposal, 0.2), xytext=(mu_current, 0.2),
                 arrowprops=dict(arrowstyle="->", lw=2.))
    ax2.set(title='likelihood(mu=%.2f) = %.2f\nlikelihood(mu=%.2f) = %.2f' % (mu_current, 1e14*likelihood_current, mu_proposal, 1e14*likelihood_proposal))

    # Posterior
    posterior_current = self.calc_posterior_analytical(mu_current)
    posterior_proposal = self.calc_posterior_analytical(mu_proposal)
    ax3.plot(x, posterior_current)
    ax3.plot([mu_current] * 2, [0, posterior_current], marker='o', color='b')
    ax3.plot([mu_proposal] * 2, [0, posterior_proposal], marker='o', color=color)
    ax3.annotate("", xy=(mu_proposal, 0.2), xytext=(mu_current, 0.2),
                 arrowprops=dict(arrowstyle="->", lw=2.))
    ax3.set(title='posterior(mu=%.2f) = %.5f\nposterior(mu=%.2f) = %.5f' % (mu_current, posterior_current, mu_proposal, posterior_proposal))

    # Trace
    if accepted:
        trace.append(mu_proposal)
    else:
        trace.append(mu_current)
    ax4.plot(trace)
    ax4.set(xlabel='iteration', ylabel='mu', title='trace')
    plt.tight_layout()

