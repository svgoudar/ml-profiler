def calculate_summary_statistics(data):
    """
    Calculate summary statistics for a given dataset.

    Parameters:
    data (DataFrame): The input dataset.

    Returns:
    dict: A dictionary containing summary statistics including mean, median, mode, 
          standard deviation, variance, minimum, and maximum values.
    """
    summary_stats = {
        'mean': data.mean(),
        'median': data.median(),
        'mode': data.mode().iloc[0],
        'std_dev': data.std(),
        'variance': data.var(),
        'min': data.min(),
        'max': data.max()
    }
    return summary_stats


def summarize_data(data):
    """
    Summarize the dataset by calculating summary statistics and displaying them.

    Parameters:
    data (DataFrame): The input dataset.
    """
    stats = calculate_summary_statistics(data)
    print("Summary Statistics:")
    for key, value in stats.items():
        print(f"{key.capitalize()}: {value}")