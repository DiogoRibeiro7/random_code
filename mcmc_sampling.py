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

        # Omitted plot details for brevity

        plt.tight_layout()
