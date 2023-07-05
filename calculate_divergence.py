from scipy.stats import norm
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


class DensityRatioModel:
    def __init__(self, kernel_bandwidth=1.0):
        self.kernel_bandwidth = kernel_bandwidth

    def fit(self, X1, X2):
        """
        Fits the density-ratio model using kernel density estimation.

        Args:
            X1 (np.ndarray): The numerator dataset.
            X2 (np.ndarray): The denominator dataset.
        """
        self.X1 = X1
        self.X2 = X2

    def predict(self, X):
        """
        Predicts the density ratio for new samples.

        Args:
            X (np.ndarray): The input dataset.

        Returns:
            np.ndarray: The density ratios for the input samples.
        """
        # Compute the numerator and denominator densities using kernel density estimation
        numerator_densities = self._kernel_density_estimate(X, self.X1)
        denominator_densities = self._kernel_density_estimate(X, self.X2)

        # Compute the density ratios
        density_ratios = numerator_densities / denominator_densities

        return density_ratios

    def _kernel_density_estimate(self, X, data):
        """
        Performs kernel density estimation.

        Args:
            X (np.ndarray): The input dataset.
            data (np.ndarray): The training dataset.

        Returns:
            np.ndarray: The estimated density values for the input samples.
        """
        n = len(data)
        d = X.shape[1]
        kernel_bandwidth = self.kernel_bandwidth

        # Compute the Gaussian kernel matrix
        K = np.exp(-np.sum((X[:, None] - data) ** 2,
                   axis=-1) / (2 * kernel_bandwidth ** 2))

        # Compute the density estimates
        density_estimates = np.sum(
            K, axis=1) / (n * (2 * np.pi * kernel_bandwidth ** 2) ** (d / 2))

        return density_estimates


# Example usage
X1 = np.random.rand(100, 2)  # Numerator dataset
X2 = np.random.rand(100, 2)  # Denominator dataset

# Create and fit the density-ratio model
model = DensityRatioModel(kernel_bandwidth=0.5)
model.fit(X1, X2)

# Predict density ratios for new samples
X_new = np.random.rand(50, 2)
density_ratios = model.predict(X_new)

print("Density Ratios:", density_ratios)


def kliep_learning(X, X_ref, kernel_func, num_iterations=100):
    """
    Performs KLIEP learning to estimate the density ratio using the KLIEP algorithm.

    Args:
        X (np.ndarray): The training dataset.
        X_ref (np.ndarray): The reference dataset.
        kernel_func (callable): Kernel function that measures the similarity between samples.
        num_iterations (int, optional): Number of iterations for the KLIEP algorithm. Defaults to 100.

    Returns:
        np.ndarray: The density ratios for the training dataset.

    Raises:
        ValueError: If the input datasets have incompatible shapes.

    Example:
        X = np.random.rand(100, 2)
        X_ref = np.random.rand(100, 2)
        kernel_func = lambda x, y: np.exp(-np.sum((x - y) ** 2) / (2 * 1.0 ** 2))
        density_ratios = kliep_learning(X, X_ref, kernel_func)
        print("Density Ratios:", density_ratios)
    """
    if X.shape[1] != X_ref.shape[1]:
        raise ValueError(
            "Input datasets must have the same number of features")

    n = X.shape[0]
    m = X_ref.shape[0]

    # Compute the kernel matrix
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            K[i, j] = kernel_func(X[i], X_ref[j])

    # Initialize the weights uniformly
    weights = np.ones(n) / n

    for _ in range(num_iterations):
        # Compute the source density estimate
        p_hat = np.dot(K, weights) / np.sum(weights)

        # Compute the importance weights
        importance_weights = np.sqrt(p_hat) / weights

        # Update the weights using importance weighting
        weights = importance_weights / np.sum(importance_weights)

    # Compute the density ratios for the training dataset
    density_ratios = np.sum(weights[:, np.newaxis] * K, axis=1)

    return density_ratios


# Example usage
X = np.random.rand(100, 2)
X_ref = np.random.rand(100, 2)
def kernel_func(x, y): return np.exp(-np.sum((x - y) ** 2) / (2 * 1.0 ** 2))


density_ratios = kliep_learning(X, X_ref, kernel_func)
print("Density Ratios:", density_ratios)


class KLIEPChangePointDetector:
    def __init__(self, kernel_bandwidth=1.0):
        self.kernel_bandwidth = kernel_bandwidth

    def detect_change_point(self, data):
        """
        Detects the change point in the given data using KLIEP.

        Args:
            data (np.ndarray): The input data.

        Returns:
            int: The index of the detected change point.

        Raises:
            ValueError: If the input data is empty or has incompatible shape.

        Example:
            data = np.random.rand(200)
            detector = KLIEPChangePointDetector()
            change_point = detector.detect_change_point(data)
            print("Change Point:", change_point)
        """
        if len(data) == 0:
            raise ValueError("Input data is empty")

        n = len(data)
        best_score = -np.inf
        change_point = None

        for i in range(1, n):
            X1 = data[:i]
            X2 = data[i:]

            # Compute KLIEP density ratios
            density_ratios = self._kliep_density_ratios(X1, X2)

            # Compute the score using KL divergence
            score = self._kl_divergence(density_ratios)

            if score > best_score:
                best_score = score
                change_point = i

        return change_point

    def _kliep_density_ratios(self, X1, X2):
        """
        Computes the KLIEP density ratios between two segments of data.

        Args:
            X1 (np.ndarray): The first segment of data.
            X2 (np.ndarray): The second segment of data.

        Returns:
            np.ndarray: The KLIEP density ratios.

        Raises:
            ValueError: If the input data has incompatible shape.
        """
        if X1.shape[0] == 0 or X2.shape[0] == 0:
            raise ValueError("Input data is empty")

        # Create and fit KLIEP density ratio model
        model = DensityRatioModel(kernel_bandwidth=self.kernel_bandwidth)
        model.fit(X1, X2)

        # Predict density ratios for X2 using the model
        density_ratios = model.predict(X2)

        return density_ratios

    def _kl_divergence(self, density_ratios):
        """
        Computes the KL divergence from the uniform distribution to the density ratios.

        Args:
            density_ratios (np.ndarray): The density ratios.

        Returns:
            float: The KL divergence.
        """
        normalized_ratios = density_ratios / np.sum(density_ratios)
        kl_divergence = np.sum(normalized_ratios * np.log(normalized_ratios))
        return kl_divergence


# Example usage
data = np.concatenate([np.random.normal(0, 1, 100),
                      np.random.normal(2, 1, 100)])
detector = KLIEPChangePointDetector(kernel_bandwidth=0.5)
change_point = detector.detect_change_point(data)
print("Change Point:", change_point)


class uLSIF:
    def __init__(self, kernel_bandwidth=1.0, regularization=0.1, num_iterations=100):
        self.kernel_bandwidth = kernel_bandwidth
        self.regularization = regularization
        self.num_iterations = num_iterations
        self.weights = None

    def fit(self, X, Y):
        """
        Fits the uLSIF model to estimate the density ratio.

        Args:
            X (np.ndarray): The numerator dataset.
            Y (np.ndarray): The denominator dataset.
        """
        K_xx = self._compute_kernel_matrix(X, X)
        K_yy = self._compute_kernel_matrix(Y, Y)
        K_xy = self._compute_kernel_matrix(X, Y)

        n = X.shape[0]
        m = Y.shape[0]

        # Initialize importance weights
        w = np.ones(n)

        for _ in range(self.num_iterations):
            K_xw = np.dot(K_xx, w)
            A = np.dot(K_xy.T, K_xw)
            B = np.dot(K_xx.T, K_xw) + self.regularization * np.eye(n)
            w_new = np.linalg.solve(B, A)
            w = w_new / np.sum(w_new)

        self.weights = w

    def predict(self, Y):
        """
        Predicts the density ratio for new samples.

        Args:
            Y (np.ndarray): The input dataset.

        Returns:
            np.ndarray: The density ratios for the input samples.
        """
        K_yx = self._compute_kernel_matrix(Y, X)
        density_ratios = (1 / X.shape[0]) * np.dot(K_yx, self.weights)
        return density_ratios

    def _compute_kernel_matrix(self, X1, X2):
        """
        Computes the Gaussian kernel matrix.

        Args:
            X1 (np.ndarray): The first dataset.
            X2 (np.ndarray): The second dataset.

        Returns:
            np.ndarray: The kernel matrix.
        """
        pairwise_distances = np.sum((X1[:, np.newaxis] - X2) ** 2, axis=-1)
        kernel_matrix = np.exp(-pairwise_distances /
                               (2 * self.kernel_bandwidth ** 2))
        return kernel_matrix


# Example usage
X = np.random.rand(100, 2)  # Numerator dataset
Y = np.random.rand(100, 2)  # Denominator dataset

# Create and fit the uLSIF model
model = uLSIF(kernel_bandwidth=0.5, regularization=0.1, num_iterations=100)
model.fit(X, Y)

# Predict density ratios for new samples
Y_new = np.random.rand(50, 2)
density_ratios = model.predict(Y_new)

print("Density Ratios:", density_ratios)


class RuLSIF:
    def __init__(self, kernel_bandwidth=1.0, regularization=0.1, num_iterations=100, reference_density=None):
        self.kernel_bandwidth = kernel_bandwidth
        self.regularization = regularization
        self.num_iterations = num_iterations
        self.weights = None
        self.reference_density = reference_density

    def fit(self, X, Y):
        """
        Fits the RuLSIF model to estimate the relative density ratio.

        Args:
            X (np.ndarray): The numerator dataset.
            Y (np.ndarray): The denominator dataset.
        """
        K_xx = self._compute_kernel_matrix(X, X)
        K_yy = self._compute_kernel_matrix(Y, Y)
        K_xy = self._compute_kernel_matrix(X, Y)

        n = X.shape[0]
        m = Y.shape[0]

        # Initialize importance weights
        w = np.ones(n)

        for _ in range(self.num_iterations):
            K_xw = np.dot(K_xx, w)
            A = np.dot(K_xy.T, K_xw)
            B = np.dot(K_xx.T, K_xw) + self.regularization * np.eye(n)
            w_new = np.linalg.solve(B, A)
            w = w_new / np.sum(w_new)

        self.weights = w

    def predict(self, Y):
        """
        Predicts the relative density ratio for new samples.

        Args:
            Y (np.ndarray): The input dataset.

        Returns:
            np.ndarray: The relative density ratios for the input samples.
        """
        K_yx = self._compute_kernel_matrix(Y, X)
        density_ratios = (1 / X.shape[0]) * np.dot(K_yx, self.weights)

        if self.reference_density is not None:
            density_ratios /= self.reference_density(Y)

        return density_ratios

    def _compute_kernel_matrix(self, X1, X2):
        """
        Computes the Gaussian kernel matrix.

        Args:
            X1 (np.ndarray): The first dataset.
            X2 (np.ndarray): The second dataset.

        Returns:
            np.ndarray: The kernel matrix.
        """
        pairwise_distances = np.sum((X1[:, np.newaxis] - X2) ** 2, axis=-1)
        kernel_matrix = np.exp(-pairwise_distances /
                               (2 * self.kernel_bandwidth ** 2))
        return kernel_matrix


# Example usage
X = np.random.rand(100, 2)  # Numerator dataset
Y = np.random.rand(100, 2)  # Denominator dataset


def reference_density(Y):
    # Define the reference density function
    return np.ones(len(Y))  # Example: Uniform reference density


# Create and fit the RuLSIF model
model = RuLSIF(kernel_bandwidth=0.5, regularization=0.1,
               num_iterations=100, reference_density=reference_density)
model.fit(X, Y)

# Predict relative density ratios for new samples
Y_new = np.random.rand(50, 2)
density_ratios = model.predict(Y_new)

print("Relative Density Ratios:", density_ratios)


"""
In this implementation, the symmetrized_pe_divergence function computes the symmetrized PE divergence between two datasets. It estimates the mean and variance of the datasets using the norm.fit function from SciPy's stats module and computes the symmetrized PE divergence based on the estimated parameters.

To demonstrate the change-point detection, an artificial time-series signal with three segments is generated, each having a different variance. The change-point locations are defined, and the symmetrized PE divergence is calculated for each potential change-point. The change-point with the maximum divergence is considered as the detected change-point.

Please note that this is a simplified example, and you may need to adapt the code based on your specific requirements and datasets.
"""






def symmetrized_pe_divergence(X, Y):
    """
    Computes the symmetrized PE divergence between two datasets.

    Args:
        X (np.ndarray): The first dataset.
        Y (np.ndarray): The second dataset.

    Returns:
        float: The symmetrized PE divergence.

    Raises:
        ValueError: If the input datasets have different lengths.
    """
    if len(X) != len(Y):
        raise ValueError("Input datasets must have the same length")

    n = len(X)

    # Estimate the mean and variance of X and Y
    mean_X, var_X = norm.fit(X)
    mean_Y, var_Y = norm.fit(Y)

    # Compute the symmetrized PE divergence
    pe_divergence_XY = (var_X + (mean_X - mean_Y)**2) / (2 * var_Y)
    pe_divergence_YX = (var_Y + (mean_Y - mean_X)**2) / (2 * var_X)
    symmetrized_pe_divergence = pe_divergence_XY + pe_divergence_YX

    return symmetrized_pe_divergence


# Generate an artificial time-series signal with three segments
segment_length = 200
signal = np.concatenate([
    np.random.normal(0, 2, segment_length),
    np.random.normal(0, 1, segment_length),
    np.random.normal(0, 2, segment_length)
])

# Define the change-point locations
change_points = [segment_length, 2 * segment_length]

# Perform change-point detection using symmetrized PE divergence
detection_results = []
for i in range(len(signal) - segment_length):
    segment1 = signal[:i+segment_length]
    segment2 = signal[i+segment_length:]
    divergence = symmetrized_pe_divergence(segment1, segment2)
    detection_results.append(divergence)

# Find the change-point with the maximum divergence
change_point = np.argmax(detection_results)
change_point += segment_length

print("Change Point:", change_point)
