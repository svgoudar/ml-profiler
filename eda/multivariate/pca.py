from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def perform_pca(data, n_components=2):
    """
    Perform Principal Component Analysis (PCA) on the given dataset.

    Parameters:
    data (pd.DataFrame): The input data for PCA.
    n_components (int): The number of principal components to return.

    Returns:
    pd.DataFrame: A DataFrame containing the principal components.
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    return pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])

def plot_pca_variance(pca):
    """
    Plot the explained variance ratio of each principal component.

    Parameters:
    pca (PCA): The PCA object fitted to the data.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
    plt.title('Explained Variance by Principal Components')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
    plt.grid()
    plt.show()

def pca_analysis(data, n_components=2):
    """
    Conduct PCA analysis and plot the explained variance.

    Parameters:
    data (pd.DataFrame): The input data for PCA.
    n_components (int): The number of principal components to return.
    """
    pca = PCA(n_components=n_components)
    pca.fit(data)
    plot_pca_variance(pca)
    return perform_pca(data, n_components)