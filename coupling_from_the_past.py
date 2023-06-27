import numpy as np

# Define the transition matrix for a biased coin (0=heads, 1=tails)
# p is the probability of getting heads
p = 0.6
transition_matrix = np.array([[p, 1-p],
                              [1-p, p]])


def flip_biased_coin(state, transition_matrix):
    return np.random.choice([0, 1], p=transition_matrix[state])


def coupling_from_the_past(transition_matrix):
    n = transition_matrix.shape[0]  # Number of states
    t = 0
    while True:
        t += 1
        states = np.arange(n)
        for _ in range(t):
            # Update each state according to the Markov chain
            states = np.array(
                [flip_biased_coin(state, transition_matrix) for state in states])
        if np.all(states == states[0]):
            # All chains have coalesced
            return states[0]


# Get a sample from the stationary distribution using CFTP
print(coupling_from_the_past(transition_matrix))


def energy_ising(spins, J=1):
    """
    Compute the energy of a configuration in the 1D Ising model.
    """
    return -J * np.sum(spins[:-1] * spins[1:])


def metropolis_hastings_step(spins, beta=1):
    """
    Perform one step of the Metropolis-Hastings algorithm for the 1D Ising model.
    """
    N = len(spins)
    i = np.random.randint(N)  # Choose a spin to flip
    delta_E = 2 * spins[i] * \
        (spins[i-1] + spins[(i+1) % N])  # Change in energy
    if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
        spins[i] *= -1  # Flip the spin
    return spins


def coupling_from_the_past(N, beta=1):
    """
    Use Coupling from the Past to sample from the stationary distribution of the 1D Ising model.
    """
    spins = np.ones(N)
    while True:
        spins_prev = spins.copy()
        for _ in range(N):
            spins = metropolis_hastings_step(spins, beta)
        if np.all(spins == spins_prev):
            return spins


# Parameters
N = 10  # Number of spins
beta = 1  # Inverse temperature

# Use CFTP to sample from the stationary distribution
spins = coupling_from_the_past(N, beta)
print(spins)


# Define the transition matrix
transition_matrix = np.array([
    [0.9, 0.1],
    [0.2, 0.8]
])

# Number of states
n_states = transition_matrix.shape[0]

# Generate some data using the Markov chain
n_data = 1000
data = np.zeros(n_data, dtype=int)
for i in range(1, n_data):
    data[i] = np.random.choice(n_states, p=transition_matrix[data[i-1]])

# Inject some outliers
outliers = np.random.randint(0, n_states, size=10)
data[np.random.choice(n_data, size=10, replace=False)] = outliers

# Now we have a dataset `data` with a few outliers
# We can try to detect the outliers using the Markov chain model

# Compute the log-probability of each observation under the model
log_probs = np.log(transition_matrix[data[:-1], data[1:]])

# Identify potential outliers as the observations with the lowest log-probabilities
n_outliers = 10
outlier_indices = np.argpartition(log_probs, n_outliers)[:n_outliers]

print("Potential outlier indices:", outlier_indices)
