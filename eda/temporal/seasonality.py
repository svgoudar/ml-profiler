def identify_seasonality(time_series_data, frequency):
    """
    Identifies seasonal patterns in the given time series data.

    Parameters:
    time_series_data (pd.Series): The time series data to analyze.
    frequency (int): The number of periods in a season.

    Returns:
    pd.Series: A series containing the seasonal component.
    """
    seasonal_decomposition = seasonal_decompose(time_series_data, model='additive', period=frequency)
    return seasonal_decomposition.seasonal


def plot_seasonality(seasonal_component):
    """
    Plots the seasonal component of the time series data.

    Parameters:
    seasonal_component (pd.Series): The seasonal component to plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(seasonal_component, label='Seasonal Component', color='blue')
    plt.title('Seasonal Component of Time Series')
    plt.xlabel('Time')
    plt.ylabel('Seasonal Value')
    plt.legend()
    plt.show()