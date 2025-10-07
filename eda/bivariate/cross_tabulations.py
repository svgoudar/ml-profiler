def create_cross_tabulation(data, column1, column2):
    """
    Create a cross-tabulation of two categorical variables.

    Parameters:
    data (DataFrame): The input DataFrame containing the data.
    column1 (str): The name of the first categorical column.
    column2 (str): The name of the second categorical column.

    Returns:
    DataFrame: A DataFrame representing the cross-tabulation.
    """
    import pandas as pd
    
    return pd.crosstab(data[column1], data[column2], margins=True, margins_name="Total")

def plot_cross_tabulation(cross_tab, title="Cross Tabulation"):
    """
    Plot the cross-tabulation as a heatmap.

    Parameters:
    cross_tab (DataFrame): The cross-tabulation DataFrame.
    title (str): The title of the plot.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(cross_tab, annot=True, fmt='g', cmap='Blues')
    plt.title(title)
    plt.xlabel(cross_tab.columns.name)
    plt.ylabel(cross_tab.index.name)
    plt.show()