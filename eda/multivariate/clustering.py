# clustering.py

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def apply_kmeans(data, n_clusters):
    """
    Apply KMeans clustering to the dataset.

    Parameters:
    data (DataFrame): The input data for clustering.
    n_clusters (int): The number of clusters to form.

    Returns:
    labels (ndarray): Cluster labels for each point in the dataset.
    kmeans (KMeans): The fitted KMeans model.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels, kmeans

def plot_clusters(data, labels, n_clusters):
    """
    Plot the clusters formed by KMeans.

    Parameters:
    data (DataFrame): The input data for clustering.
    labels (ndarray): Cluster labels for each point in the dataset.
    n_clusters (int): The number of clusters.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
    plt.title(f'KMeans Clustering with {n_clusters} Clusters')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.colorbar(label='Cluster Label')
    plt.show()

def optimal_number_of_clusters(data, max_clusters=10):
    """
    Determine the optimal number of clusters using the silhouette score.

    Parameters:
    data (DataFrame): The input data for clustering.
    max_clusters (int): The maximum number of clusters to test.

    Returns:
    best_n_clusters (int): The optimal number of clusters.
    """
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        labels, _ = apply_kmeans(data, n_clusters)
        silhouette_avg = silhouette_score(data, labels)
        silhouette_scores.append(silhouette_avg)
    
    best_n_clusters = np.argmax(silhouette_scores) + 2  # +2 because range starts at 2
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(2, max_clusters + 1))
    plt.grid()
    plt.show()
    
    return best_n_clusters