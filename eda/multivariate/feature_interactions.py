def analyze_feature_interactions(data, features):
    """
    Analyze interactions between multiple features in the dataset.

    Parameters:
    data (DataFrame): The input dataset.
    features (list): List of feature names to analyze interactions.

    Returns:
    DataFrame: A DataFrame containing interaction terms and their effects.
    """
    from sklearn.preprocessing import PolynomialFeatures
    import pandas as pd

    # Create interaction terms
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    interaction_terms = poly.fit_transform(data[features])

    # Create a DataFrame for interaction terms
    interaction_df = pd.DataFrame(interaction_terms, columns=poly.get_feature_names_out(features))

    return interaction_df


def visualize_interactions(interaction_df):
    """
    Visualize the interactions between features.

    Parameters:
    interaction_df (DataFrame): DataFrame containing interaction terms.

    Returns:
    None: Displays the interaction plots.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Plotting interaction terms
    plt.figure(figsize=(10, 6))
    sns.heatmap(interaction_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Interaction Heatmap')
    plt.show()