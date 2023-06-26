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
