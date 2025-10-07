def load_data(file_path):
    """Load data from a specified file path."""
    import pandas as pd
    return pd.read_csv(file_path)

def save_data(data, file_path):
    """Save data to a specified file path."""
    data.to_csv(file_path, index=False)

def display_head(data, n=5):
    """Display the first n rows of the DataFrame."""
    return data.head(n)

def check_nulls(data):
    """Check for null values in the DataFrame."""
    return data.isnull().sum()

def get_data_types(data):
    """Get the data types of each column in the DataFrame."""
    return data.dtypes

def describe_data(data):
    """Get a summary of the DataFrame."""
    return data.describe()