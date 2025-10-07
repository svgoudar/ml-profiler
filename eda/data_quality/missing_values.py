def identify_missing_values(data):
    """
    Identifies missing values in the dataset.

    Parameters:
    data (DataFrame): The input dataset.

    Returns:
    DataFrame: A DataFrame indicating the number of missing values per column.
    """
    return data.isnull().sum()


def handle_missing_values(data, method='drop', fill_value=None):
    """
    Handles missing values in the dataset based on the specified method.

    Parameters:
    data (DataFrame): The input dataset.
    method (str): The method to handle missing values ('drop', 'fill').
    fill_value: The value to fill missing entries if method is 'fill'.

    Returns:
    DataFrame: The dataset with missing values handled.
    """
    if method == 'drop':
        return data.dropna()
    elif method == 'fill':
        return data.fillna(fill_value)
    else:
        raise ValueError("Method must be 'drop' or 'fill'.")