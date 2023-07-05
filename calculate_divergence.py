import numpy as np


def calculate_divergence(X1: np.ndarray, X2: np.ndarray) -> float:
    """
    Calculates the divergence-based dissimilarity measure between two datasets.

    Args:
        X1 (np.ndarray): The first dataset.
        X2 (np.ndarray): The second dataset.

    Returns:
        float: The divergence-based dissimilarity measure.

    Raises:
        ValueError: If the input datasets have incompatible shapes.

    Example:
        X1 = np.random.rand(100, 2)
        X2 = np.random.rand(100, 2)
        divergence = calculate_divergence(X1, X2)
        print("Divergence:", divergence)
    """
    if X1.shape[1] != X2.shape[1]:
        raise ValueError(
            "Input datasets must have the same number of features")

    # Calculate pairwise distances between samples
    distances = np.sqrt(np.sum((X1[:, None] - X2) ** 2, axis=-1))

    # Calculate the mean distances along the rows and columns
    row_mean = np.mean(distances, axis=1)
    col_mean = np.mean(distances, axis=0)

    # Calculate the Kullback-Leibler divergence
    divergence = np.sum(row_mean * np.log(row_mean / col_mean))

    return divergence


# Example usage
X1 = np.random.rand(100, 2)
X2 = np.random.rand(100, 2)
divergence = calculate_divergence(X1, X2)
print("Divergence:", divergence)


def kliep(X, Y, kernel_bandwidth=1.0, num_iterations=100):
    """
    Performs direct density-ratio estimation using the KLIEP algorithm.

    Args:
        X (np.ndarray): The source dataset.
        Y (np.ndarray): The target dataset.
        kernel_bandwidth (float, optional): Bandwidth parameter for the Gaussian kernel. Defaults to 1.0.
        num_iterations (int, optional): Number of iterations for the KLIEP algorithm. Defaults to 100.

    Returns:
        np.ndarray: The weights for each sample in the source dataset.

    Raises:
        ValueError: If the input datasets have incompatible shapes.

    Example:
        X = np.random.rand(100, 2)
        Y = np.random.rand(100, 2)
        weights = kliep(X, Y)
        print("Weights:", weights)
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            "Input datasets must have the same number of features")

    n = X.shape[0]
    m = Y.shape[0]

    # Compute the Gaussian kernel matrix
    K = np.exp(-np.sum((X[:, None] - Y) ** 2, axis=-1) /
               (2 * kernel_bandwidth ** 2))

    # Initialize the weights uniformly
    weights = np.ones(n) / n

    for _ in range(num_iterations):
        # Compute the source density estimate
        p_hat = np.dot(K, weights) / np.sum(weights)

        # Compute the importance weights
        importance_weights = np.divide(np.sqrt(p_hat), weights)

        # Update the weights using importance weighting
        weights = np.divide(importance_weights, np.sum(importance_weights))

    return weights


# Example usage
X = np.random.rand(100, 2)
Y = np.random.rand(100, 2)
weights = kliep(X, Y)
print("Weights:", weights)
