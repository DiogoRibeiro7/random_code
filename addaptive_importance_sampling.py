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


from typing import Callable, List, Tuple
import numpy as np

def gradient_adaptive_population_importance_sampling(
    target_distribution: Callable[[np.ndarray], float],
    proposal_distributions: List[Tuple[Callable[[np.ndarray], float], np.ndarray]],
    num_samples: int,
    num_iterations: int,
    step_size: float = 0.1,
    repulsion_strength: float = 0.1
) -> List[np.ndarray]:
    """
    Performs Gradient Adaptive Population Importance Sampling (GAPIS) to generate samples from the target distribution.

    GAPIS adapts N proposals by adjusting the location and scale parameters using the gradient ascent of the target
    and the Hessian of the target, respectively. An advanced implementation adds repulsive interaction among proposals
    to promote cooperative exploration of the target.

    Args:
        target_distribution: The target distribution to generate samples from.
        proposal_distributions: A list of tuples, where each tuple contains a proposal distribution
                                and its initial location parameters.
        num_samples: The total number of samples to generate.
        num_iterations: The number of iterations to perform.
        step_size: The step size for the gradient ascent and Hessian update.
        repulsion_strength: The strength of the repulsive interaction among proposals.

    Returns:
        A list of samples generated using GAPIS.
    """
    # Check if the number of samples and iterations are valid
    if num_samples <= 0 or num_iterations <= 0:
        raise ValueError("Number of samples and iterations must be greater than zero.")

    num_proposals = len(proposal_distributions)
    samples = []

    for _ in range(num_iterations):
        # Generate K samples from each proposal distribution
        K = int(num_samples / (num_iterations * num_proposals))
        for proposal, location in proposal_distributions:
            proposal_samples = np.random.randn(K, len(location)) * location
            proposal_weights = target_distribution(proposal_samples) / proposal(proposal_samples)
            proposal_weights /= np.sum(proposal_weights)
            samples.extend(proposal_samples)

        # Update the location and scale parameters of the proposals
        for i in range(num_proposals):
            proposal, location = proposal_distributions[i]

            # Gradient ascent of the target for location update
            gradient = compute_gradient(target_distribution, location)
            location += step_size * gradient

            # Hessian update for scale parameter
            hessian = compute_hessian(target_distribution, location)
            scale = np.linalg.inv(hessian)

            proposal_distributions[i] = (proposal, location, scale)

        # Apply repulsion among proposals
        for i in range(num_proposals):
            for j in range(num_proposals):
                if i != j:
                    repulsion_direction = proposal_distributions[i][1] - proposal_distributions[j][1]
                    repulsion_norm = np.linalg.norm(repulsion_direction)
                    repulsion_force = repulsion_strength / repulsion_norm**3 if repulsion_norm > 0 else 0
                    proposal_distributions[i][1] += repulsion_force * repulsion_direction

    return samples


def compute_gradient(func: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    """
    Computes the gradient of a scalar-valued function at a given point using central differences.

    Args:
        func: The function to compute the gradient of.
        x: The point at which to compute the gradient.

    Returns:
        The gradient of the function at the given point.
    """
    epsilon = 1e-6
    gradient = np.zeros_like(x)

    for i in range(len(x)):
        delta = np.zeros_like(x)
        delta[i] = epsilon

        gradient[i] = (func(x + delta) - func(x - delta)) / (2 * epsilon)

    return gradient


def compute_hessian(func: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    """
    Computes the Hessian matrix of a scalar-valued function at a given point using central differences.

    Args:
        func: The function to compute the Hessian matrix of.
        x: The point at which to compute the Hessian matrix.

    Returns:
        The Hessian matrix of the function at the given point.
    """
    epsilon = 1e-6
    n = len(x)
    hessian = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            delta_i = np.zeros_like(x)
            delta_j = np.zeros_like(x)
            delta_i[i] = epsilon
            delta_j[j] = epsilon

            hessian[i, j] = (func(x + delta_i + delta_j) - func(x + delta_i - delta_j) -
                             func(x - delta_i + delta_j) + func(x - delta_i - delta_j)) / (4 * epsilon**2)

    return hessian


# Example usage
def target_distribution(x: np.ndarray) -> float:
    """
    Example target distribution (unnormalized).

    Args:
        x: The input value.

    Returns:
        The unnormalized probability density function of the target distribution.
    """
    return np.exp(-np.sum(x**2))


def proposal_distribution_1(x: np.ndarray) -> float:
    """
    Example proposal distribution 1.

    Args:
        x: The input value.

    Returns:
        The probability density function of proposal distribution 1.
    """
    return np.exp(-np.sum((x - 1)**2) / 2) / (2 * np.pi)


def proposal_distribution_2(x: np.ndarray) -> float:
    """
    Example proposal distribution 2.

    Args:
        x: The input value.

    Returns:
        The probability density function of proposal distribution 2.
    """
    return np.exp(-np.sum((x + 1)**2) / 2) / (2 * np.pi)


proposal_distributions = [
    (proposal_distribution_1, np.array([1, 1])),
    (proposal_distribution_2, np.array([-1, -1]))
]

num_samples = 1000
num_iterations = 10

samples = gradient_adaptive_population_importance_sampling(
    target_distribution, proposal_distributions, num_samples, num_iterations)

# Print the estimated mean
estimated_mean = np.mean(samples, axis=0)
print("Estimated mean:", estimated_mean)



import numpy as np

def streaming_importance_sampling(target_distribution, proposal_distribution, num_samples):
    """
    Performs Streaming Importance Sampling to estimate properties of a target distribution.

    Args:
        target_distribution: A function that returns the unnormalized probability density of the target distribution.
        proposal_distribution: A function that returns the probability density of the proposal distribution.
        num_samples: The number of samples to use for the estimation.

    Returns:
        The estimated mean and variance of the target distribution.
    """
    # Initialize variables
    weighted_samples = []
    weights = []

    # Generate samples and calculate weights
    for _ in range(num_samples):
        sample = proposal_distribution()
        weight = target_distribution(sample) / proposal_distribution(sample)
        weighted_samples.append(sample * weight)
        weights.append(weight)

    # Normalize weights
    weights = np.array(weights) / np.sum(weights)

    # Estimate mean and variance
    estimated_mean = np.sum(weighted_samples, axis=0) / np.sum(weights)
    estimated_variance = np.sum(weights * (np.array(weighted_samples) - estimated_mean) ** 2, axis=0)

    return estimated_mean, estimated_variance


# Example usage
def target_distribution(x):
    """Example target distribution (unnormalized)."""
    return np.exp(-x ** 2)

def proposal_distribution():
    """Example proposal distribution."""
    return np.random.normal(0, 1)

num_samples = 1000
estimated_mean, estimated_variance = streaming_importance_sampling(target_distribution, proposal_distribution, num_samples)

print("Estimated mean:", estimated_mean)
print("Estimated variance:", estimated_variance)


import numpy as np

def gaussian_particle_filter(initial_state, num_particles, transition_model, measurement_model, observations):
    """
    Performs Gaussian Particle Filtering for state estimation in a dynamic system.

    Args:
        initial_state: The initial state of the system.
        num_particles: The number of particles to use.
        transition_model: A function that models the state transition.
        measurement_model: A function that models the measurement process.
        observations: The sequence of measurements.

    Returns:
        The estimated sequence of states.
    """
    num_timesteps = len(observations)
    state_dim = initial_state.shape[0]
    state_particles = np.zeros((num_timesteps, num_particles, state_dim))

    # Initialize particles at the initial state
    state_particles[0] = np.random.multivariate_normal(initial_state, np.eye(state_dim), size=num_particles)

    for t in range(1, num_timesteps):
        # Resampling step
        weights = np.zeros(num_particles)
        for i in range(num_particles):
            state_particles[t, i] = transition_model(state_particles[t-1, i])
            weights[i] = measurement_model(observations[t], state_particles[t, i])

        # Normalize weights
        weights /= np.sum(weights)

        # Resample particles
        resampled_indices = np.random.choice(np.arange(num_particles), size=num_particles, replace=True, p=weights)
        state_particles[t] = state_particles[t, resampled_indices]

    return state_particles

# Example usage
def transition_model(state):
    """Example state transition model."""
    return np.random.multivariate_normal(state, np.eye(state.shape[0]))

def measurement_model(observation, state):
    """Example measurement model."""
    return np.random.normal(state, 1)

initial_state = np.array([0, 0])
num_particles = 100
observations = [1, 2, 3, 4]

estimated_states = gaussian_particle_filter(initial_state, num_particles, transition_model, measurement_model, observations)

print("Estimated states:")
for t in range(len(observations)):
    print(f"Time step {t+1}: {estimated_states[t]}")

