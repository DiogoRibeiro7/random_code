from typing import List
import numpy as np

def bootstrap_outlier_detection(data: np.ndarray, num_resamples: int, alpha: float) -> List[int]:
    """
    Detect outliers in a data set using bootstrap resampling.

    Parameters
    ----------
    data : np.ndarray
        The data set.
    num_resamples : int
        The number of bootstrap resamples to create.
    alpha : float
        The significance level for the confidence interval (e.g., 0.05 for a 95% confidence interval).

    Returns
    -------
    List[int]
        The indices of the detected outliers.
    """

    # Create bootstrap resamples and calculate their means
    bootstrap_means = np.array([np.mean(np.random.choice(data, size=len(data))) for _ in range(num_resamples)])

    # Calculate the confidence interval for the resampled means
    lower_bound = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper_bound = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    # Detect outliers as data points that fall outside the confidence interval
    outliers = np.where((data < lower_bound) | (data > upper_bound))

    return outliers[0].tolist()

# Test the function with an example
data = np.array([1, 2, 3, 4, 5, 100])  # Here, 100 is an outlier
print("Detected outliers: ", bootstrap_outlier_detection(data, num_resamples=1000, alpha=0.05))
