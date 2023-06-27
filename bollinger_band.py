import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_bollinger_bands(data, window_size, num_std):
    """
    Calculate Bollinger Bands for a given data series.

    Args:
        data (numpy.ndarray or pandas.Series): The time series data.
        window_size (int): The size of the moving window for calculating the rolling mean and standard deviation.
        num_std (float): The number of standard deviations to use for the upper and lower bands.

    Returns:
        pandas.DataFrame: A DataFrame containing the original data, rolling mean, upper band, and lower band.
    """
    if not isinstance(data, (np.ndarray, pd.Series)):
        raise ValueError("Data must be a numpy array or pandas Series.")

    # Convert data to numeric type
    data = pd.to_numeric(data, errors="coerce")

    rolling_mean = data.rolling(window=window_size).mean()
    rolling_std = data.rolling(window=window_size).std()

    upper_band = rolling_mean + (num_std * rolling_std)
    lower_band = rolling_mean - (num_std * rolling_std)

    bollinger_bands = pd.DataFrame({'Data': data, 'Rolling Mean': rolling_mean,
                                    'Upper Band': upper_band, 'Lower Band': lower_band})

    return bollinger_bands

# Example usage
# Assuming heartbeats per minute data is stored in a pandas Series called 'heartbeats'
heartbeats = pd.Series([...])  # Replace [...] with your actual data

window_size = 20  # Size of the moving window for calculating the rolling mean and standard deviation
num_std = 2  # Number of standard deviations to use for the upper and lower bands

bollinger_bands = calculate_bollinger_bands(heartbeats, window_size, num_std)

# Plotting the data and Bollinger Bands
plt.figure(figsize=(10, 6))
plt.plot(bollinger_bands.index, bollinger_bands['Data'], label='Heartbeats per Minute')
plt.plot(bollinger_bands.index, bollinger_bands['Rolling Mean'], label='Rolling Mean')
plt.plot(bollinger_bands.index, bollinger_bands['Upper Band'], label='Upper Band')
plt.plot(bollinger_bands.index, bollinger_bands['Lower Band'], label='Lower Band')
plt.legend(loc='best')
plt.xlabel('Time')
plt.ylabel('Heartbeats per Minute')
plt.title('Bollinger Bands')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_box_plots(data, window_size):
    """
    Plot box plots for each window of data.

    Args:
        data (numpy.ndarray or list): The heartbeats per minute data.
        window_size (int): The size of the window for each box plot.

    Returns:
        None
    """
    if not isinstance(data, (np.ndarray, list)):
        raise ValueError("Data must be a numpy array or list.")

    num_windows = len(data) // window_size  # Number of windows to create

    # Create an array of x-axis values representing time
    time = np.arange(len(data))

    # Create subplots for box plots
    fig, ax = plt.subplots(num_windows, figsize=(10, 6))

    # Generate box plots for each window
    for i in range(num_windows):
        start = i * window_size  # Start index of the window
        end = start + window_size  # End index of the window
        window_data = data[start:end]  # Data within the window
        window_time = time[start:end]  # Time values for the window

        # Plot box plot for the window
        ax[i].boxplot(window_data)

        # Customize the plot for each window
        ax[i].set_xlabel('Time')
        ax[i].set_ylabel('Heartbeats per Minute')
        ax[i].set_xticks(np.arange(0, len(window_time), 10))  # Set x-axis tick locations
        ax[i].set_xticklabels(window_time[::10])  # Set x-axis tick labels for every 10th time point

    # Set the overall title for the plot
    fig.suptitle('Heart BPM Box Plots by Window')

    # Adjust layout spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example usage
# Assuming heartbeats per minute data is stored in a numpy array called 'heartbeats'
heartbeats = np.random.randint(low=60, high=100, size=1000)  # Replace with your actual heartbeats per minute data

window_size = 50  # Size of the window for each box plot

plot_box_plots(heartbeats, window_size)
