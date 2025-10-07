def analyze_categorical(data, column):
    """
    Analyzes a categorical variable by providing frequency counts and visualizations.

    Parameters:
    data (DataFrame): The input data containing the categorical variable.
    column (str): The name of the categorical column to analyze.

    Returns:
    dict: A dictionary containing frequency counts and visualizations.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Frequency counts
    frequency_counts = data[column].value_counts()

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=column, order=frequency_counts.index)
    plt.title(f'Frequency Counts of {column}')
    plt.xticks(rotation=45)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    return {
        'frequency_counts': frequency_counts,
        'visualization': f'Visualization for {column} displayed.'
    }