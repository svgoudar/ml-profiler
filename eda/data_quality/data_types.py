def check_data_type(data):
    """Check the data type of the input data."""
    return type(data)

def convert_data_type(data, target_type):
    """Convert the input data to the specified target type."""
    try:
        return target_type(data)
    except ValueError:
        raise ValueError(f"Cannot convert {data} to {target_type}")

def is_numeric(data):
    """Check if the input data is numeric."""
    return isinstance(data, (int, float))

def is_categorical(data):
    """Check if the input data is categorical."""
    return isinstance(data, str) or isinstance(data, bool)

def is_datetime(data):
    """Check if the input data is a datetime object."""
    return isinstance(data, (datetime.date, datetime.datetime))