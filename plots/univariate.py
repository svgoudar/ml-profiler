import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from profiler.plots import EDABuilder


class UnivariateBuilder(EDABuilder):
    """Child class for univariate plots."""
    
    _plots = {}  # Optional: keep subclass-specific plots
    
    def __init__(self, df):
        super().__init__(df)
        # Register all built-in univariate plots automatically
        self.register_default_plots()
    
    def register_default_plots(self):
        """Register common univariate plots"""
        self.register_plot(self.histogram_plot)
        self.register_plot(self.box_plot)
        self.register_plot(self.kde_plot)
    
    # ----------------------
    # Define univariate plots
    # ----------------------
    def histogram_plot(self, columns=None, figsize=(12,6), plot_kwargs=None):
        columns = self._numeric_cols(columns)
        plot_kwargs = plot_kwargs or {}
        n_cols = len(columns)
        if n_cols == 0:
            print("No numeric columns to plot.")
            return
        
        n_rows = int(np.ceil(n_cols / 2))
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(columns):
            sns.histplot(self.df[col], kde=True, ax=axes[i], **plot_kwargs)
            axes[i].set_title(f"Histogram of {col}")
        
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()
    
    def box_plot(self, columns=None, figsize=(12,6), plot_kwargs=None):
        columns = self._numeric_cols(columns)
        plot_kwargs = plot_kwargs or {}
        n_cols = len(columns)
        if n_cols == 0:
            print("No numeric columns to plot.")
            return

        n_rows = int(np.ceil(n_cols / 2))
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(columns):
            sns.boxplot(y=self.df[col], ax=axes[i], **plot_kwargs)
            axes[i].set_title(f"Box Plot of {col}")
        
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()
    
    def kde_plot(self, columns=None, figsize=(12,6), plot_kwargs=None):
        columns = self._numeric_cols(columns)
        plot_kwargs = plot_kwargs or {}
        n_cols = len(columns)
        if n_cols == 0:
            print("No numeric columns to plot.")
            return

        n_rows = int(np.ceil(n_cols / 2))
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(columns):
            sns.kdeplot(self.df[col], ax=axes[i], **plot_kwargs)
            axes[i].set_title(f"KDE Plot of {col}")
        
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()
