import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


def random_variate(pdf, n=1000, xmin=0, xmax=1):
    """
    Rejection method for random number generation.
    Uses the rejection method for generating random numbers derived from an arbitrary   
    probability distribution.

    Parameters:
        pdf (function): Probability distribution function from which to generate random numbers.
        n (int): Desired number of random values. Default is 1000.
        xmin (float): Lower range of random numbers desired. Default is 0.
        xmax (float): Upper range of random numbers desired. Default is 1.

    Returns:
        ran (numpy.array): Array of N random variates that follow the input pdf.
        n_trials (int): Number of trials the function needed to achieve N.
    """
    # Create an array over the interval (xmin, xmax)
    x = np.linspace(xmin, xmax, 1000)
    y = pdf(x)

    pmin = 0.0
    pmax = y.max()

    # Initialize counters
    n_accept = 0
    n_trials = 0

    # Output list of random numbers
    ran = []

    # Keep generating numbers until desired count n is reached
    while n_accept < n:
        # Draw a sample from uniform distribution (xmin, xmax)
        x = np.random.uniform(xmin, xmax)
        # Draw a sample from uniform distribution (pmin, pmax)
        y = np.random.uniform(pmin, pmax)

        # If y is less than the pdf at that point, accept x
        if y < pdf(x):
            ran.append(x)
            n_accept += 1
        n_trials += 1

    ran = np.asarray(ran)

    return ran, n_trials


def gaussian_pdf(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)


random_numbers, n_trials = random_variate(
    gaussian_pdf, n=1000, xmin=-3, xmax=3)

# Let's print the first 10 numbers to see what they look like
print(random_numbers[:10])


def inverse_transform_method(lmbda=1, size=1):
    uniform_samples = np.random.uniform(size=size)
    exponential_samples = -np.log(1 - uniform_samples) / lmbda
    return exponential_samples


def acceptance_rejection_method(size=1):
    samples = []
    while len(samples) < size:
        x = np.random.uniform(-4, 4)  # Propose from a uniform distribution
        u = np.random.uniform(0, 0.4)  # Uniform vertical proposal
        if u <= stats.norm.pdf(x):  # Acceptance criterion
            samples.append(x)
    return np.array(samples)


def box_muller_transform(size=1):
    u1 = np.random.uniform(size=size)
    u2 = np.random.uniform(size=size)
    R = np.sqrt(-2 * np.log(u1))
    theta = 2 * np.pi * u2
    z1 = R * np.cos(theta)
    z2 = R * np.sin(theta)
    return z1, z2  # Both are standard normal variables


def marsaglia_polar_method(size=1):
    z1 = []
    z2 = []
    while len(z1) < size:
        u = np.random.uniform(-1, 1)
        v = np.random.uniform(-1, 1)
        s = u * u + v * v
        if 0 < s <= 1:
            z1.append(u * np.sqrt(-2 * np.log(s) / s))
            z2.append(v * np.sqrt(-2 * np.log(s) / s))
    return np.array(z1), np.array(z2)  # Both are standard normal variables


def box_muller_transform(size=1):
    u1 = np.random.uniform(size=size)
    u2 = np.random.uniform(size=size)
    R = np.sqrt(-2 * np.log(u1))
    theta = 2 * np.pi * u2
    z1 = R * np.cos(theta)
    z2 = R * np.sin(theta)
    return z1, z2  # Both are standard normal variables


# Generate 10000 random variables
z1, z2 = box_muller_transform(size=10000)

# Plot histogram for z1
plt.hist(z1, bins=30, density=True, alpha=0.6, color='g')
plt.title('Histogram of Generated Random Variables (z1)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for z2
plt.hist(z2, bins=30, density=True, alpha=0.6, color='g')
plt.title('Histogram of Generated Random Variables (z2)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
