import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from profiler.plots import EDABuilder


class MultivariateBuilder(EDABuilder):
    """Child class for multivariate plots (correlation, pairplot, radar, heatmaps)."""

    _plots = {}  # subclass-specific plot registry

    def __init__(self, df):
        super().__init__(df)
        self.register_default_plots()

    def register_default_plots(self):
        """Register multivariate plots"""
        self.register_plot(self.corr_heatmap)
        self.register_plot(self.pair_plot)
        self.register_plot(self.radar_plot)

    # ----------------------
    # Correlation Heatmap
    # ----------------------
    def corr_heatmap(self, columns=None, figsize=(10, 8), plot_kwargs=None):
        columns = self._numeric_cols(columns)
        if not columns:
            print("No numeric columns to plot.")
            return
        plot_kwargs = plot_kwargs or {}
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
        if not columns:
            print("No numeric columns to plot.")
            return
        plot_kwargs = plot_kwargs or {}
        sns.pairplot(self.df[columns], hue=hue, **plot_kwargs)
        plt.show()

    # ----------------------
    # Radar / Spider plot
    # ----------------------
    def radar_plot(self, variable, columns=None, figsize=(8, 8), plot_kwargs=None):
        columns = self._numeric_cols(columns)
        if variable not in self.df.columns:
            print(f"Variable '{variable}' not found in dataframe.")
            return
        plot_kwargs = plot_kwargs or {}

        # Correlations of variable with others
        corr_matrix = self.df[columns].corr()
        corrs = corr_matrix[variable].drop(variable)

        angles = np.linspace(0, 2 * np.pi, len(corrs), endpoint=False).tolist()
        corrs_list = corrs.values.tolist()

        # Close the radar
        angles += angles[:1]
        corrs_list += corrs_list[:1]

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))
        ax.plot(angles, corrs_list, "o-", linewidth=2, **plot_kwargs)
        ax.fill(angles, corrs_list, alpha=0.25, **plot_kwargs)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(corrs.index)
        ax.set_ylim(-1, 1)
        ax.set_title(f"Radar Plot for {variable}", pad=20)
        ax.grid(True)
        plt.show()
