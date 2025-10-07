import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from profiler.plots import EDABuilder


class ClassificationEDA(EDABuilder):
    _plots = {}

    def __init__(self, df, target):
        super().__init__(df)
        self.target = target
        self.register_default_plots()

    def register_default_plots(self):
        self.register_plot(self.target_distribution)
        self.register_plot(self.numeric_vs_target_box)
        self.register_plot(self.corr_heatmap)

    # --------------------------
    # Plots
    # --------------------------
    def target_distribution(self, figsize=(6, 4), plot_kwargs=None):
        plot_kwargs = plot_kwargs or {}
        plt.figure(figsize=figsize)
        sns.countplot(x=self.target, data=self.df, **plot_kwargs)
        plt.title(f"Target Distribution: {self.target}")
        plt.show()

    def numeric_vs_target_box(
        self, numeric_cols=None, figsize=(12, 6), plot_kwargs=None
    ):
        numeric_cols = self._numeric_cols(numeric_cols)
        plot_kwargs = plot_kwargs or {}
        n_cols = len(numeric_cols)
        n_rows = int(np.ceil(n_cols / 2))
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            sns.boxplot(
                x=self.df[self.target], y=self.df[col], ax=axes[i], **plot_kwargs
            )
            axes[i].set_title(f"{col} by {self.target}")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.show()


class RegressionEDA(EDABuilder):
    _plots = {}  # subclass-specific plot registry

    def __init__(self, df, target):
        super().__init__(df)
        self.target = target
        self.register_default_plots()

    def register_default_plots(self):
        """Register regression-specific plots"""
        self.register_plot(self.target_distribution)
        self.register_plot(self.numeric_vs_target_scatter)
        self.register_plot(self.residual_plot)
        self.register_plot(self.corr_heatmap)
        self.register_plot(self.pair_plot)

    # ----------------------
    # Target distribution
    # ----------------------
    def target_distribution(self, figsize=(6, 4), plot_kwargs=None):
        plot_kwargs = plot_kwargs or {}
        plt.figure(figsize=figsize)
        sns.histplot(self.df[self.target], kde=True, **plot_kwargs)
        plt.title(f"Target Distribution: {self.target}")
        plt.show()

    # ----------------------
    # Numeric vs target scatter plots
    # ----------------------
    def numeric_vs_target_scatter(
        self, numeric_cols=None, figsize=(12, 6), plot_kwargs=None
    ):
        numeric_cols = self._numeric_cols(numeric_cols)
        plot_kwargs = plot_kwargs or {}
        n_cols = len(numeric_cols)
        if n_cols == 0:
            print("No numeric columns to plot.")
            return

        n_rows = int(np.ceil(n_cols / 2))
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.scatterplot(
                x=self.df[col], y=self.df[self.target], ax=axes[i], **plot_kwargs
            )
            axes[i].set_title(f"{col} vs {self.target}")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.show()

    # ----------------------
    # Residual plot (requires predicted values)
    # ----------------------
    def residual_plot(self, y_true=None, y_pred=None, figsize=(6, 4), plot_kwargs=None):
        plot_kwargs = plot_kwargs or {}
        if y_true is None or y_pred is None:
            print("y_true and y_pred are required for residual plot.")
            return
        residuals = y_true - y_pred
        plt.figure(figsize=figsize)
        sns.scatterplot(x=y_pred, y=residuals, **plot_kwargs)
        plt.axhline(0, color="r", linestyle="--")
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.show()

    # ----------------------
    # Correlation heatmap
    # ----------------------
    def corr_heatmap(self, columns=None, figsize=(10, 8), plot_kwargs=None):
        columns = self._numeric_cols(columns)
        plot_kwargs = plot_kwargs or {}
        if not columns:
            print("No numeric columns to plot.")
            return
        corr = self.df[columns].corr()
        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", **plot_kwargs)
        plt.title("Correlation Heatmap")
        plt.show()

    # ----------------------
    # Pairplot for numeric columns
    # ----------------------
    def pair_plot(self, columns=None, hue=None, plot_kwargs=None):
        columns = self._numeric_cols(columns)
        plot_kwargs = plot_kwargs or {}
        if not columns:
            print("No numeric columns to plot.")
            return
        sns.pairplot(self.df[columns], hue=hue, **plot_kwargs)
        plt.show()


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class ClusteringEDA(EDABuilder):
    """Child class for clustering / unsupervised EDA."""

    _plots = {}

    def __init__(self, df, cluster_col):
        """
        df : DataFrame
        cluster_col : column containing cluster labels
        """
        super().__init__(df)
        self.cluster_col = cluster_col
        self.register_default_plots()

    def register_default_plots(self):
        """Register clustering-specific plots"""
        self.register_plot(self.cluster_boxplot)
        self.register_plot(self.cluster_violinplot)
        self.register_plot(self.corr_heatmap)
        self.register_plot(self.pca_plot)
        self.register_plot(self.pair_plot)

    # ----------------------
    # Boxplot by cluster
    # ----------------------
    def cluster_boxplot(self, numeric_cols=None, figsize=(12, 6), plot_kwargs=None):
        numeric_cols = self._numeric_cols(numeric_cols)
        plot_kwargs = plot_kwargs or {}
        n_cols = len(numeric_cols)
        if n_cols == 0:
            print("No numeric columns to plot.")
            return

        n_rows = int(np.ceil(n_cols / 2))
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.boxplot(
                x=self.cluster_col, y=col, data=self.df, ax=axes[i], **plot_kwargs
            )
            axes[i].set_title(f"{col} by {self.cluster_col}")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.show()

    # ----------------------
    # Violin plot by cluster
    # ----------------------
    def cluster_violinplot(self, numeric_cols=None, figsize=(12, 6), plot_kwargs=None):
        numeric_cols = self._numeric_cols(numeric_cols)
        plot_kwargs = plot_kwargs or {}
        n_cols = len(numeric_cols)
        if n_cols == 0:
            print("No numeric columns to plot.")
            return

        n_rows = int(np.ceil(n_cols / 2))
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.violinplot(
                x=self.cluster_col, y=col, data=self.df, ax=axes[i], **plot_kwargs
            )
            axes[i].set_title(f"{col} by {self.cluster_col}")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.show()

    # ----------------------
    # Correlation heatmap
    # ----------------------
    def corr_heatmap(self, columns=None, figsize=(10, 8), plot_kwargs=None):
        columns = self._numeric_cols(columns)
        plot_kwargs = plot_kwargs or {}
        if not columns:
            print("No numeric columns to plot.")
            return
        corr = self.df[columns].corr()
        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", **plot_kwargs)
        plt.title("Correlation Heatmap")
        plt.show()

    # ----------------------
    # PCA scatter plot colored by cluster
    # ----------------------
    def pca_plot(self, components=2, figsize=(8, 6), plot_kwargs=None):
        plot_kwargs = plot_kwargs or {}
        numeric_cols = self._numeric_cols()
        X = self.df[numeric_cols].values
        pca = PCA(n_components=components)
        X_pca = pca.fit_transform(X)
        df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(components)])
        df_pca[self.cluster_col] = self.df[self.cluster_col].values

        plt.figure(figsize=figsize)
        if components == 2:
            sns.scatterplot(
                x="PC1", y="PC2", hue=self.cluster_col, data=df_pca, **plot_kwargs
            )
        elif components == 3:
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
            for cluster in df_pca[self.cluster_col].unique():
                cluster_data = df_pca[df_pca[self.cluster_col] == cluster]
                ax.scatter(
                    cluster_data["PC1"],
                    cluster_data["PC2"],
                    cluster_data["PC3"],
                    label=f"Cluster {cluster}",
                    **plot_kwargs,
                )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            plt.legend()
        plt.title("PCA Plot by Cluster")
        plt.show()

    # ----------------------
    # Pairplot colored by cluster
    # ----------------------
    def pair_plot(self, columns=None, figsize=(10, 10), plot_kwargs=None):
        columns = self._numeric_cols(columns)
        plot_kwargs = plot_kwargs or {}
        if not columns:
            print("No numeric columns to plot.")
            return
        sns.pairplot(
            self.df[columns + [self.cluster_col]], hue=self.cluster_col, **plot_kwargs
        )
        plt.show()
