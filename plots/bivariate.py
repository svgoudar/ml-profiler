import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from types import MethodType

class BivariateBuilder(EDABuilder):
    """Child class for bivariate plots (numeric vs numeric, numeric vs categorical, categorical vs categorical)."""
    
    _plots = {}  # subclass-specific plot registry
    
    def __init__(self, df):
        super().__init__(df)
        self.register_default_plots()
    
    def register_default_plots(self):
        """Register common bivariate plots"""
        self.register_plot(self.scatter_plot)
        self.register_plot(self.box_plot)
        self.register_plot(self.violin_plot)
        self.register_plot(self.countplot_by_category)
    
    # ----------------------
    # Bivariate numeric vs numeric
    # ----------------------
    def scatter_plot(self, x, y, hue=None, figsize=(7,5), plot_kwargs=None):
        plot_kwargs = plot_kwargs or {}
        plt.figure(figsize=figsize)
        sns.scatterplot(data=self.df, x=x, y=y, hue=hue, **plot_kwargs)
        plt.title(f"Scatter Plot: {x} vs {y}")
        plt.show()
    
    # ----------------------
    # Numeric vs categorical
    # ----------------------
    def box_plot(self, numeric_col, categorical_col, figsize=(8,5), plot_kwargs=None):
        plot_kwargs = plot_kwargs or {}
        plt.figure(figsize=figsize)
        sns.boxplot(x=categorical_col, y=numeric_col, data=self.df, **plot_kwargs)
        plt.title(f"Box Plot: {numeric_col} by {categorical_col}")
        plt.show()
    
    def violin_plot(self, numeric_col, categorical_col, figsize=(8,5), plot_kwargs=None):
        plot_kwargs = plot_kwargs or {}
        plt.figure(figsize=figsize)
        sns.violinplot(x=categorical_col, y=numeric_col, data=self.df, **plot_kwargs)
        plt.title(f"Violin Plot: {numeric_col} by {categorical_col}")
        plt.show()
    
    # ----------------------
    # Categorical vs categorical
    # ----------------------
    def countplot_by_category(self, col1, col2=None, figsize=(8,5), plot_kwargs=None):
        """If col2 is given, hue=col2, else simple countplot"""
        plot_kwargs = plot_kwargs or {}
        plt.figure(figsize=figsize)
        sns.countplot(x=col1, hue=col2, data=self.df, **plot_kwargs)
        title = f"Count Plot of {col1}" + (f" by {col2}" if col2 else "")
        plt.title(title)
        plt.show()
