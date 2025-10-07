# Contents of `c:\Github\ml_model_profiler\app\eda\temporal\time_series.py`

def analyze_time_series(data, time_column, value_column):
    """
    Analyzes time series data by calculating basic statistics and plotting the series.

    Parameters:
    - data: DataFrame containing the time series data.
    - time_column: The name of the column containing time information.
    - value_column: The name of the column containing the values to analyze.

    Returns:
    - A dictionary with basic statistics of the time series.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # Ensure the time column is in datetime format
    data[time_column] = pd.to_datetime(data[time_column])

    # Set the time column as the index
    data.set_index(time_column, inplace=True)

    # Calculate basic statistics
    stats = {
        'mean': data[value_column].mean(),
        'median': data[value_column].median(),
        'std_dev': data[value_column].std(),
        'min': data[value_column].min(),
        'max': data[value_column].max(),
    }

    # Plot the time series
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[value_column], label=value_column)
    plt.title('Time Series Analysis')
    plt.xlabel('Time')
    plt.ylabel(value_column)
    plt.legend()
    plt.grid()
    plt.show()

    return stats

def seasonal_decompose(data, time_column, value_column):
    """
    Decomposes the time series into trend, seasonal, and residual components.

    Parameters:
    - data: DataFrame containing the time series data.
    - time_column: The name of the column containing time information.
    - value_column: The name of the column containing the values to analyze.

    Returns:
    - A seasonal decomposition object.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    # Ensure the time column is in datetime format
    data[time_column] = pd.to_datetime(data[time_column])
    data.set_index(time_column, inplace=True)

    # Decompose the time series
    decomposition = seasonal_decompose(data[value_column], model='additive')
    return decomposition

def plot_seasonal_decomposition(decomposition):
    """
    Plots the seasonal decomposition components.

    Parameters:
    - decomposition: A seasonal decomposition object.
    """
    plt.figure(figsize=(12, 8))
    decomposition.plot()
    plt.show()