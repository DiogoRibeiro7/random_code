import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_whisker(data):
    # Create a DataFrame and reset the index to get dates and values as separate columns
    df = data.reset_index()
    df.columns = ['Date', 'Value']
    df['Date'] = df['Date'].dt.date  # Convert datetime to date (without time)
    
    # Create a boxplot
    plt.figure(figsize=(10, 5))  # Adjust the figure size as needed
    sns.boxplot(x='Date', y='Value', data=df)
    plt.title('Boxplot by Date')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility if needed
    plt.tight_layout()  # Adjust layout for better visibility
    plt.show()

# Create a sample time series dataset
time_steps = pd.date_range(start='2022-01-01', periods=240, freq='H')  # Now generating a new value every hour for 10 days
values = np.random.normal(0, 1, 240)  # Random values for the time steps
data = pd.Series(values, index=time_steps)

# Call the plot_whisker function
plot_whisker(data)
