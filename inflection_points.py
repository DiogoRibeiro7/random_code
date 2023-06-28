import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d

def find_inflexion_points(y_vals: np.array, x_vals: np.array, smoothing_sigma=2) -> list:
    """
    Function to find inflection points of the given data.

    Args:
        y_vals (numpy array): y-values of the data points.
        x_vals (numpy array): x-values of the data points.
        smoothing_sigma (int): Sigma value for Gaussian smoothing.

    Returns:
        list: List of x-values where the function has inflection points.
    """
    if not isinstance(y_vals, np.ndarray) or not isinstance(x_vals, np.ndarray):
        raise ValueError("The inputs should be numpy arrays")

    # Smooth the data
    y_vals_smooth = gaussian_filter1d(y_vals, smoothing_sigma)

    # First derivative
    f_prime = np.gradient(y_vals_smooth, x_vals)

    # Second derivative
    f_double_prime = np.gradient(f_prime, x_vals)

    # Inflection points are where the second derivative has local maxima or minima
    inflexion_points_indices = np.concatenate([argrelextrema(f_double_prime, np.greater)[0], argrelextrema(f_double_prime, np.less)[0]])

    inflexion_points = x_vals[inflexion_points_indices]

    return inflexion_points

def plot_function_and_inflexion_points(y_vals, x_vals, inflexion_points):
    """
    Function to plot the function and its inflection points.

    Args:
        y_vals (numpy array): y-values of the data points.
        x_vals (numpy array): x-values of the data points.
        inflexion_points (list): List of x-values where the function has inflection points.

    """
    if not isinstance(y_vals, np.ndarray) or not isinstance(x_vals, np.ndarray) or not isinstance(inflexion_points, np.ndarray):
        raise ValueError("The inputs should be numpy arrays")

    try:
        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label="Function")
        plt.plot(inflexion_points, np.interp(inflexion_points, x_vals, y_vals), 'ro')  # Plot inflection points in red
        plt.title("Function and its inflection points")
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"An error occurred while plotting: {e}")
        
        


# Example usage with some noise
x_vals = np.linspace(-10, 10, 1000)
y_vals = (x_vals**3 - 3*x_vals**2 + 2) + np.random.normal(0, 10, len(x_vals))

smoothing_sigma = calculate_sigma(y_vals)

inflexion_points = find_inflexion_points(y_vals, x_vals, smoothing_sigma)
plot_function_and_inflexion_points(y_vals, x_vals, inflexion_points)

