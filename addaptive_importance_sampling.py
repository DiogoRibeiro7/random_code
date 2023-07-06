from typing import Callable, List
import numpy as np

def adaptive_importance_sampling(
    target_distribution: Callable[[float], float],
    proposal_distribution: Callable[[float], float],
    proposal_sample: Callable[[], float],
    num_samples: int
) -> List[float]:
    """
    Performs adaptive importance sampling to estimate the mean of a target distribution.

    Args:
        target_distribution: The target distribution to estimate the mean of.
        proposal_distribution: The proposal distribution used for sampling.
        proposal_sample: A function that generates samples from the proposal distribution.
        num_samples: The number of samples to generate.

    Returns:
        A list of samples generated using adaptive importance sampling.
    """
    # Check if the number of samples is valid
    if num_samples <= 0:
        raise ValueError("Number of samples must be greater than zero.")

    samples = []
    weights = []

    for _ in range(num_samples):
        # Generate a sample from the proposal distribution
        sample = proposal_sample()

        # Calculate the weight for the sample
        weight = target_distribution(sample) / proposal_distribution(sample)
        weights.append(weight)
        samples.append(sample)

    # Normalize the weights
    weights = np.array(weights) / sum(weights)

    return samples


# Example usage
def target_distribution(x: float) -> float:
    """
    Example target distribution (unnormalized).

    Args:
        x: The input value.

    Returns:
        The unnormalized probability density function of the target distribution.
    """
    return np.exp(-x**2)


def proposal_distribution(x: float) -> float:
    """
    Example proposal distribution.

    Args:
        x: The input value.

    Returns:
        The probability density function of the proposal distribution.
    """
    return 1 / np.sqrt(2*np.pi) * np.exp(-x**2/2)


def proposal_sample() -> float:
    """
    Example function to generate samples from the proposal distribution.

    Returns:
        A sample generated from the proposal distribution.
    """
    return np.random.normal(0, 1)


num_samples = 1000
samples = adaptive_importance_sampling(target_distribution, proposal_distribution, proposal_sample, num_samples)

# Print the estimated mean
estimated_mean = np.mean(samples)
print("Estimated mean:", estimated_mean)



from typing import Callable, List, Tuple
import numpy as np

def mixture_population_monte_carlo(
    target_distribution: Callable[[float], float],
    proposal_distributions: List[Tuple[Callable[[float], float], float]],
    num_samples: int,
    num_iterations: int
) -> List[float]:
    """
    Performs Mixture Population Monte Carlo (M-PMC) to generate samples from the target distribution.

    M-PMC uses a mixture of proposal distributions to generate samples at each iteration,
    and adapts the mixture to decrease the Kullback-Leibler divergence between the mixture and the target.

    Args:
        target_distribution: The target distribution to generate samples from.
        proposal_distributions: A list of tuples, where each tuple contains a proposal distribution
                                and its initial weight. The proposal distribution is a function that takes
                                a float as input and returns the probability density function of the proposal.
        num_samples: The total number of samples to generate.
        num_iterations: The number of iterations to perform.

    Returns:
        A list of samples generated using M-PMC.
    """
    # Check if the number of samples and iterations are valid
    if num_samples <= 0 or num_iterations <= 0:
        raise ValueError("Number of samples and iterations must be greater than zero.")

    num_proposals = len(proposal_distributions)
    samples = []

    for _ in range(num_iterations):
        # Generate K samples from the current mixture of proposal distributions
        K = int(num_samples / num_iterations)
        proposal_samples = []
        proposal_weights = []

        for i in range(num_proposals):
            proposal, weight = proposal_distributions[i]
            proposal_samples.extend(np.random.choice(proposal_samples, size=K, replace=True))
            proposal_weights.extend([weight] * K)

        proposal_samples = np.array(proposal_samples)
        proposal_weights = np.array(proposal_weights)

        # Calculate the importance weights for the proposal samples
        importance_weights = target_distribution(proposal_samples) / np.sum(proposal_weights * proposal_samples)

        # Normalize the importance weights
        normalized_weights = importance_weights / np.sum(importance_weights)

        # Resample K samples according to the importance weights
        resampled_indices = np.random.choice(np.arange(len(proposal_samples)), size=K, replace=True, p=normalized_weights)
        resampled_samples = proposal_samples[resampled_indices]

        samples.extend(resampled_samples)

        # Update the proposal distributions based on the resampled samples
        for i in range(num_proposals):
            proposal, weight = proposal_distributions[i]
            proposal_weights[i] = np.sum(importance_weights * (proposal_samples == proposal_samples[i])) / K

    return samples


# Example usage
def target_distribution(x: float) -> float:
    """
    Example target distribution (unnormalized).

    Args:
        x: The input value.

    Returns:
        The unnormalized probability density function of the target distribution.
    """
    return np.exp(-x**2)


def proposal_distribution_1(x: float) -> float:
    """
    Example proposal distribution 1.

    Args:
        x: The input value.

    Returns:
        The probability density function of proposal distribution 1.
    """
    return np.exp(-((x - 1) ** 2) / 2) / np.sqrt(2 * np.pi)


def proposal_distribution_2(x: float) -> float:
    """
    Example proposal distribution 2.

    Args:
        x: The input value.

    Returns:
        The probability density function of proposal distribution 2.
    """
    return np.exp(-((x + 1) ** 2) / 2) / np.sqrt(2 * np.pi)


proposal_distributions = [
    (proposal_distribution_1, 0.5),
    (proposal_distribution_2, 0.5)
]

num_samples = 1000
num_iterations = 10

samples = mixture_population_monte_carlo(target_distribution, proposal_distributions, num_samples, num_iterations)

# Print the estimated mean
estimated_mean = np.mean(samples)
print("Estimated mean:", estimated_mean)



from typing import Callable, List, Tuple
import numpy as np

def nonlinear_population_monte_carlo(
    target_distribution: Callable[[float], float],
    proposal_distribution: Callable[[float, float], float],
    num_samples: int,
    num_iterations: int
) -> List[float]:
    """
    Performs Nonlinear Population Monte Carlo (N-PMC) to generate samples from the target distribution.

    N-PMC computes standard importance weights using the proposal distribution,
    and then applies a nonlinear transformation to reduce weight variance.
    The transformed weights are used for the adaptation step, where a Gaussian proposal
    is employed with adapted mean vector and covariance matrix.

    Args:
        target_distribution: The target distribution to generate samples from.
        proposal_distribution: The proposal distribution used for importance weights and adaptation.
        num_samples: The total number of samples to generate.
        num_iterations: The number of iterations to perform.

    Returns:
        A list of samples generated using N-PMC.
    """
    # Check if the number of samples and iterations are valid
    if num_samples <= 0 or num_iterations <= 0:
        raise ValueError("Number of samples and iterations must be greater than zero.")

    samples = []

    for _ in range(num_iterations):
        # Generate K samples from the proposal distribution
        K = int(num_samples / num_iterations)
        proposal_samples = np.random.randn(K)
        importance_weights = target_distribution(proposal_samples) / proposal_distribution(proposal_samples)

        # Nonlinear transformation of the weights
        transformed_weights = importance_weights ** 2 / np.sum(importance_weights ** 2)

        # Resample K samples according to the transformed weights
        resampled_indices = np.random.choice(np.arange(K), size=K, replace=True, p=transformed_weights)
        resampled_samples = proposal_samples[resampled_indices]

        samples.extend(resampled_samples)

        # Adapt the proposal distribution using resampled samples
        adapted_mean = np.mean(resampled_samples)
        adapted_covariance = np.cov(resampled_samples)

        proposal_distribution = lambda x, mu=adapted_mean, cov=adapted_covariance: multivariate_normal(x, mu, cov)

    return samples


# Example usage
def target_distribution(x: float) -> float:
    """
    Example target distribution (unnormalized).

    Args:
        x: The input value.

    Returns:
        The unnormalized probability density function of the target distribution.
    """
    return np.exp(-x**2)


def proposal_distribution(x: float, mu: float, cov: float) -> float:
    """
    Example proposal distribution.

    Args:
        x: The input value.
        mu: The mean of the proposal distribution.
        cov: The covariance matrix of the proposal distribution.

    Returns:
        The probability density function of the proposal distribution.
    """
    return multivariate_normal(x, mu, cov)


def multivariate_normal(x: float, mu: float, cov: float) -> float:
    """
    Multivariate normal distribution.

    Args:
        x: The input value.
        mu: The mean of the distribution.
        cov: The covariance matrix of the distribution.

    Returns:
        The probability density function of the multivariate normal distribution.
    """
    d = len(x)
    normalization = 1 / (np.sqrt((2 * np.pi) ** d * np.linalg.det(cov)))
    exponent = -0.5 * np.dot(np.dot((x - mu).T, np.linalg.inv(cov)), (x - mu))
    return normalization * np.exp(exponent)


num_samples = 1000
num_iterations = 10

samples = nonlinear_population_monte_carlo(target_distribution, proposal_distribution, num_samples, num_iterations)

# Print the estimated mean
estimated_mean = np.mean(samples)
print("Estimated mean:", estimated_mean)


from typing import Callable, List, Tuple
import numpy as np

def deterministic_mixture_population_monte_carlo(
    target_distribution: Callable[[float], float],
    proposal_distributions: List[Callable[[float], float]],
    num_samples: int,
    num_iterations: int
) -> List[float]:
    """
    Performs Deterministic Mixture Population Monte Carlo (DM-PMC) to generate samples from the target distribution.

    DM-PMC calculates weights using a modified equation and generates K samples per each proposal.
    It reduces the variance of the estimators and promotes the replication of proposals in relevant parts of the target.
    At each iteration, the population of KN samples is reduced to N via either global or local resampling.

    Args:
        target_distribution: The target distribution to generate samples from.
        proposal_distributions: A list of proposal distributions.
        num_samples: The total number of samples to generate.
        num_iterations: The number of iterations to perform.

    Returns:
        A list of samples generated using DM-PMC.
    """
    # Check if the number of samples and iterations are valid
    if num_samples <= 0 or num_iterations <= 0:
        raise ValueError("Number of samples and iterations must be greater than zero.")

    num_proposals = len(proposal_distributions)
    samples = []

    for _ in range(num_iterations):
        # Generate K samples from each proposal distribution
        K = int(num_samples / (num_iterations * num_proposals))
        for proposal in proposal_distributions:
            proposal_samples = np.random.choice(proposal_samples, size=K, replace=True)
            weights = target_distribution(proposal_samples) / proposal(proposal_samples)
            weights /= np.sum(weights)
            samples.extend(proposal_samples)

        # Resample N samples from the generated KN samples
        N = len(proposal_distributions)
        resampled_indices = np.random.choice(np.arange(len(samples)), size=N, replace=True, p=weights)
        resampled_samples = [samples[i] for i in resampled_indices]
        samples = resampled_samples

    return samples


# Example usage
def target_distribution(x: float) -> float:
    """
    Example target distribution (unnormalized).

    Args:
        x: The input value.

    Returns:
        The unnormalized probability density function of the target distribution.
    """
    return np.exp(-x**2)


def proposal_distribution_1(x: float) -> float:
    """
    Example proposal distribution 1.

    Args:
        x: The input value.

    Returns:
        The probability density function of proposal distribution 1.
    """
    return np.exp(-((x - 1) ** 2) / 2) / np.sqrt(2 * np.pi)


def proposal_distribution_2(x: float) -> float:
    """
    Example proposal distribution 2.

    Args:
        x: The input value.

    Returns:
        The probability density function of proposal distribution 2.
    """
    return np.exp(-((x + 1) ** 2) / 2) / np.sqrt(2 * np.pi)


proposal_distributions = [proposal_distribution_1, proposal_distribution_2]

num_samples = 1000
num_iterations = 10

samples = deterministic_mixture_population_monte_carlo(target_distribution, proposal_distributions, num_samples, num_iterations)

# Print the estimated mean
estimated_mean = np.mean(samples)
print("Estimated mean:", estimated_mean)
