import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import (
    pointbiserialr,
    f_oneway,
    pearsonr,
    spearmanr,
    kendalltau,
    chi2_contingency,
)


class StatAnalyzer:
    """Advanced OOP class for statistical feature analysis across ML problems."""

    def __init__(self, df, target=None, problem_type=None):
        """
        df : pd.DataFrame
        target : column name of target (None for unsupervised)
        problem_type : 'classification', 'regression', 'clustering'
        """
        self.df = df
        self.target = target
        self.problem_type = problem_type
        self.results = pd.DataFrame()
        self._plots = {}

    # ----------------------
    # Column utilities
    # ----------------------
    def _numeric_cols(self, cols=None):
        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        if cols is None:
            return numeric_cols
        return [c for c in cols if c in numeric_cols]

    def _categorical_cols(self, cols=None):
        cat_cols = self.df.select_dtypes(include="object").columns.tolist()
        if cols is None:
            return cat_cols
        return [c for c in cat_cols if c in cat_cols]

    # ----------------------
    # Feature selection / statistical tests
    # ----------------------
    def f_test_classification(self, k="all", visualize=False):
        """ANOVA F-test for numeric features vs categorical target"""
        if self.problem_type != "classification":
            raise ValueError("F-test is only for classification problems.")
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        numeric_cols = self._numeric_cols()
        X_num = X[numeric_cols]
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_num, y)
        f_scores = selector.scores_
        p_values = selector.pvalues_
        self.results = pd.DataFrame(
            {"Feature": numeric_cols, "F-value": f_scores, "p-value": p_values}
        )
        self.results.query("p-value < 0.05").sort_values(
            by="F-value", ascending=False, inplace=True
        )

        return self.results

    def point_biserial_classification(self):
        """Point-biserial correlation for binary target and numeric features"""
        if self.problem_type != "classification":
            raise ValueError("Point-biserial only applies to classification.")
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        numeric_cols = self._numeric_cols()
        res = {}
        for col in numeric_cols:
            corr, _ = pointbiserialr(X[col], y)
            res[col] = corr
        self.results = pd.DataFrame(
            {"Feature": list(res.keys()), "PointBiserialCorr": list(res.values())}
        )
        return self.results.sort_values(
            by="PointBiserialCorr", key=abs, ascending=False
        )

    def cramers_v(self, feature, target=None):
        """Cramér's V correlation for categorical vs categorical"""
        target = target or self.target
        confusion_matrix = pd.crosstab(self.df[feature], self.df[target])
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min(kcorr - 1, rcorr - 1))

    def correlation_regression(self, method="pearson"):
        """Compute correlation of numeric features with numeric target"""
        if self.problem_type != "regression":
            raise ValueError(
                "Correlation regression only applies to regression problems."
            )
        target_series = self.df[self.target]
        numeric_cols = self._numeric_cols()
        res = {}
        for col in numeric_cols:
            corr = target_series.corr(self.df[col], method=method)
            res[col] = corr
        self.results = pd.DataFrame(
            {"Feature": list(res.keys()), f"{method}_corr": list(res.values())}
        )
        return self.results.sort_values(by=f"{method}_corr", key=abs, ascending=False)

    # ----------------------
    # Dynamic plot registration
    # ----------------------
    def register_plot(self, func):
        """Register new plot dynamically"""
        self._plots[func.__name__] = func
        return func

    def add_plot_method(self, name, func):
        from types import MethodType

        setattr(self, name, MethodType(func, self))
        self._plots[name] = getattr(self, name)

    def __getattr__(self, name):
        if name in self._plots:
            return self._plots[name]
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

    # ----------------------
    # Example plot: barplot for feature importance
    # ----------------------
    def barplot_top_features(
        self, score_col, top_n=10, figsize=(8, 6), plot_kwargs=None
    ):
        plot_kwargs = plot_kwargs or {}
        if self.results.empty:
            raise ValueError("No results available. Run a statistical test first.")
        df_plot = self.results.head(top_n)
        plt.figure(figsize=figsize)
        sns.barplot(x=score_col, y="Feature", data=df_plot, **plot_kwargs)
        plt.title(f"Top {top_n} Features by {score_col}")
        plt.show()
