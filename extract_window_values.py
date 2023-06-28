import numpy as np
import pandas as pd


def extract_window_values(data, window):
    # Create an empty DataFrame to store the extracted values
    extracted_data = pd.DataFrame(columns=['Open', 'Close', 'Low', 'High',"Volume"])

    # Iterate over the data in the specified window size
    for i in range(0, len(data), window):
        window_data = data.iloc[i:i+window]  # Extract the data within the window

        # Extract the first, last, lowest, and highest values within the window
        first_value = window_data['TimeSeries'].iloc[0]
        last_value = window_data['TimeSeries'].iloc[-1]
        lowest_value = window_data['TimeSeries'].min()
        highest_value = window_data['TimeSeries'].max()
        volume = len(window_data['TimeSeries'])
        # Append the extracted values to the DataFrame
        extracted_data.loc[i] = [first_value, last_value, lowest_value, highest_value, volume]

    return extracted_data


import matplotlib.pyplot as plt
from matplotlib.dates import num2date

def plot_candlestick(extracted_data):
    # Convert the index to matplotlib dates
    dates = num2date(extracted_data.index.to_numpy())

    # Prepare the data for candlestick plot
    candlestick_data = zip(dates, extracted_data['Open'], extracted_data['Close'], extracted_data['Low'], extracted_data['High'])

    # Plot the candlestick chart
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title('Candlestick Chart')

    for date, first, last, lowest, highest in candlestick_data:
        ax.plot([date, date], [lowest, highest], color='black')
        ax.plot([date, date], [first, last], color='red' if first > last else 'green')

    # Format x-axis as dates
    ax.xaxis_date()
    ax.autoscale_view()

    # Set labels and grid
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True)

    plt.show()




# Define the number of data points
num_points = 1440

# Generate Unix timestamps for time
timestamps = pd.date_range(
    start='2022-01-01', periods=num_points, freq='min').astype(int) // 10**9

# Generate time series data with normal distribution
mean = 80
std = 20
time_series = np.random.normal(mean, std, num_points)

# Create the DataFrame
data = pd.DataFrame({'Timestamp': timestamps, 'TimeSeries': time_series})

# Set the 'Timestamp' column as the index
data.set_index('Timestamp', inplace=True)

# Display the DataFrame
print(data)


# Assuming you have generated the 'data' DataFrame with Unix timestamps and time series data as shown in the previous code example

window_size = 60  # Specify the window size

extracted_data = extract_window_values(data, window_size)
print(extracted_data)

plot_candlestick(extracted_data)
