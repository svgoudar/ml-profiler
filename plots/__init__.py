import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from types import MethodType


class EDABuilder:
    """EDA Builder with decorator-based dynamic plot registration."""

    _plots = {}  # Class-level registry

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}  # Store analysis results

    # -----------------------------
    # Magic methods
    # -----------------------------
    def __getitem__(self, key):
        return self.results.get(key, None)

    def __setitem__(self, key, value):
        self.results[key] = value

    def __getattr__(self, name):
        """Access registered plots as attributes"""
        if name in self._plots:
            return MethodType(self._plots[name], self)
        raise AttributeError(f"'EDABuilder' has no attribute '{name}'")

    def __call__(self, plot_name, *args, **kwargs):
        """Call plot by name"""
        if plot_name in self._plots:
            return self._plots[plot_name](self, *args, **kwargs)
        raise ValueError(f"Plot '{plot_name}' not registered!")

    # -----------------------------
    # Decorator for registration
    # -----------------------------
    @classmethod
    def register_plot(cls, func):
        """Decorator to register a plot"""
        cls._plots[func.__name__] = func
        return func

    # -----------------------------
    # Helper methods
    # -----------------------------
    def _numeric_cols(self, cols=None):
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if cols is None:
            return numeric_cols
        return [c for c in cols if c in numeric_cols]

    def _categorical_cols(self, cols=None):
        cat_cols = self.df.select_dtypes(include="object").columns.tolist()
        if cols is None:
            return cat_cols
        return [c for c in cols if c in cat_cols]
