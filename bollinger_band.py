import pandas as pd
import matplotlib.pyplot as plt

def calculate_bollinger_bands(prices, window=5):
    # Convert prices to a pandas Series
    prices_series = pd.Series(prices)

    # Calculate the moving average
    rolling_mean = prices_series.rolling(window).mean()

    # Calculate the standard deviation
    rolling_std = prices_series.rolling(window).std()

    # Calculate the upper and lower Bollinger Bands
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)

    return prices_series, rolling_mean, upper_band, lower_band

def plot_bollinger_bands(prices, window=5):
    prices_series, rolling_mean, upper_band, lower_band = calculate_bollinger_bands(prices, window)

    # Plot the Bollinger Bands
    plt.figure(figsize=(10, 6))
    plt.plot(prices_series, label='Price')
    plt.plot(rolling_mean, label='Moving Average', linestyle='--')
    plt.plot(upper_band, label='Upper Bollinger Band')
    plt.plot(lower_band, label='Lower Bollinger Band')
    plt.title('Bollinger Bands')
    plt.xlabel('Period')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

prices = [85.05, 84.92, 85.57, 85.97, 86.23, 87.17, 87.77, 87.29, 86.74, 87.1,
          87.67, 88.36, 88.25, 88.63, 89.37, 89.79, 89.31, 88.58, 89.53, 89.66]

plot_bollinger_bands(prices)


import pandas as pd
import mplfinance as mpf

def calculate_bollinger_bands(df, window=5):
    # Calculate the moving average
    df['MA'] = df['Close'].rolling(window).mean()

    # Calculate the standard deviation
    df['Std'] = df['Close'].rolling(window).std()

    # Calculate the upper and lower Bollinger Bands
    df['Upper'] = df['MA'] + (2 * df['Std'])
    df['Lower'] = df['MA'] - (2 * df['Std'])

    return df

def plot_candlestick_with_bollinger(df, window=5):
    df = calculate_bollinger_bands(df, window)

    # Ensure index is of type DatetimeIndex
    df.index = pd.to_datetime(df.index)

    # Plot the candlestick chart with Bollinger Bands
    mpf.plot(df, type='candle', style='charles', title='Candlestick Chart with Bollinger Bands',
             ylabel='Price', addplot=[mpf.make_addplot(df['Upper']), mpf.make_addplot(df['Lower'])])

# Example input data with "Open", "High", "Low", and "Close" columns
data = {'Date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05'],
        'Open': [85.10, 84.90, 85.50, 85.80, 86.20],
        'High': [85.50, 85.00, 86.00, 86.20, 86.50],
        'Low': [84.80, 84.50, 85.00, 85.70, 85.80],
        'Close': [85.05, 84.92, 85.57, 85.97, 86.23]}

# Convert input data to a DataFrame
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

plot_candlestick_with_bollinger(df)
