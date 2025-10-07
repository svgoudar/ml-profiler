def detect_duplicates(data):
    """
    Detect duplicate entries in the dataset.

    Parameters:
    data (DataFrame): The input dataset.

    Returns:
    DataFrame: A DataFrame containing the duplicate entries.
    """
    duplicates = data[data.duplicated()]
    return duplicates


def remove_duplicates(data):
    """
    Remove duplicate entries from the dataset.

    Parameters:
    data (DataFrame): The input dataset.

    Returns:
    DataFrame: A DataFrame with duplicates removed.
    """
    cleaned_data = data.drop_duplicates()
    return cleaned_data


def report_duplicates(data):
    """
    Report the number of duplicate entries in the dataset.

    Parameters:
    data (DataFrame): The input dataset.

    Returns:
    int: The number of duplicate entries.
    """
    return data.duplicated().sum()