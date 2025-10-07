import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict, Any, Tuple
import functools
import hashlib
import pickle
from contextlib import contextmanager


class CachedAnalysisMeta(type):
    """
    Metaclass that adds caching functionality to analysis classes.
    """
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Add cache storage to the class
        namespace['_cache'] = {}
        namespace['_cache_enabled'] = True
        
        # Wrap methods with caching decorator
        for attr_name, attr_value in namespace.items():
            if callable(attr_value) and not attr_name.startswith('_') and attr_name not in ['clear_cache', 'enable_cache', 'disable_cache']:
                namespace[attr_name] = mcs._add_caching(attr_value)
        
        return super().__new__(mcs, name, bases, namespace)
    
    @staticmethod
    def _add_caching(func):
        """Add caching functionality to a method."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not getattr(self, '_cache_enabled', True):
                return func(self, *args, **kwargs)
            
            # Create cache key from function name and arguments
            cache_key = CachedAnalysisMeta._create_cache_key(func.__name__, args, kwargs)
            
            # Check if result is cached
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            # Compute and cache result
            result = func(self, *args, **kwargs)
            self._cache[cache_key] = result
            return result
        
        return wrapper
    
    @staticmethod
    def _create_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
        """Create a unique cache key from function arguments."""
        cache_data = {
            'func_name': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        cache_str = str(cache_data)
        return hashlib.md5(cache_str.encode()).hexdigest()


class BaseEDAAnalyzer(ABC, metaclass=CachedAnalysisMeta):
    """
    Abstract base class for all EDA analyzers with advanced magic methods.
    """
    
    def __init__(self, data: pd.DataFrame):
        """Initialize the analyzer with a dataset."""
        self.data = data
        self.numeric_data = data.select_dtypes(include=[np.number])
        self._results = {}
        self._auto_execute = True
        self._validate_data()
    
    def __call__(self, auto_plot: bool = True, **kwargs) -> 'BaseEDAAnalyzer':
        """Make the analyzer callable to execute default analysis."""
        self.compute_analysis(**kwargs)
        if auto_plot:
            self.plot_analysis(**kwargs)
        return self
    
    def __enter__(self) -> 'BaseEDAAnalyzer':
        """Context manager entry."""
        self._auto_execute = False
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - auto execute analysis."""
        if not exc_type:  # No exception occurred
            self()
    
    def __repr__(self) -> str:
        """String representation of the analyzer."""
        return f"{self.__class__.__name__}(data_shape={self.data.shape}, cache_size={len(self._cache)})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        summary = self.get_summary()
        return f"""
{self.__class__.__name__} Analysis:
- Data Shape: {summary.get('data_shape', 'Unknown')}
- Numeric Columns: {len(summary.get('numeric_columns', []))}
- Cache Entries: {len(self._cache)}
- Results Available: {list(self._results.keys())}
        """.strip()
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to results."""
        if key in self._results:
            return self._results[key]
        raise KeyError(f"Result '{key}' not found. Available: {list(self._results.keys())}")
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-like assignment to results."""
        self._results[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if result exists."""
        return key in self._results
    
    def __len__(self) -> int:
        """Return number of cached results."""
        return len(self._results)
    
    def __iter__(self):
        """Iterate over result keys."""
        return iter(self._results.keys())
    
    def __bool__(self) -> bool:
        """Return True if analyzer has computed results."""
        return bool(self._results)
    
    def __add__(self, other: 'BaseEDAAnalyzer') -> Dict[str, Any]:
        """Combine results from two analyzers."""
        if not isinstance(other, BaseEDAAnalyzer):
            raise TypeError("Can only add BaseEDAAnalyzer instances")
        return {**self._results, **other._results}
    
    def __rshift__(self, method_name: str) -> Any:
        """Chain method calls using >> operator."""
        if hasattr(self, method_name):
            return getattr(self, method_name)()
        raise AttributeError(f"Method '{method_name}' not found")
    
    @property
    def results(self) -> Dict[str, Any]:
        """Get all computed results."""
        return self._results.copy()
    
    @contextmanager
    def no_cache(self):
        """Context manager to temporarily disable caching."""
        old_state = self._cache_enabled
        self.disable_cache()
        try:
            yield self
        finally:
            self._cache_enabled = old_state
    
    def _validate_data(self) -> None:
        """Validate the input data."""
        if self.data.empty:
            raise ValueError("Input data cannot be empty")
        if self.numeric_data.empty:
            raise ValueError("No numeric columns found in the data")
    
    @abstractmethod
    def compute_analysis(self, **kwargs) -> Any:
        """Abstract method to compute the main analysis."""
        pass
    
    @abstractmethod
    def plot_analysis(self, **kwargs) -> None:
        """Abstract method to plot the analysis results."""
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the analysis."""
        return {
            'data_shape': self.data.shape,
            'numeric_columns': list(self.numeric_data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'results_count': len(self._results)
        }
    
    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self._cache.clear()
    
    def enable_cache(self) -> None:
        """Enable caching for this analyzer."""
        self._cache_enabled = True
    
    def disable_cache(self) -> None:
        """Disable caching for this analyzer."""
        self._cache_enabled = False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache."""
        return {
            'cache_size': len(self._cache),
            'cache_enabled': self._cache_enabled,
            'cached_methods': list(self._cache.keys())
        }


class CorrelationAnalyzer(BaseEDAAnalyzer):
    """
    Correlation analysis implementation with advanced magic methods.
    """
    
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.correlation_matrix = None
    
    def compute_analysis(self, method: str = 'pearson', columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Compute the correlation matrix for the dataset."""
        data_to_analyze = self.numeric_data[columns] if columns else self.numeric_data
        self.correlation_matrix = data_to_analyze.corr(method=method)
        self['correlation_matrix'] = self.correlation_matrix
        self['method'] = method
        return self.correlation_matrix
    
    def plot_analysis(self, title: str = 'Correlation Matrix', cmap: str = 'coolwarm', 
                     figsize: tuple = (10, 8), annot: bool = True) -> None:
        """Plot the correlation matrix as a heatmap."""
        if self.correlation_matrix is None:
            self.compute_analysis()
            
        plt.figure(figsize=figsize)
        sns.heatmap(self.correlation_matrix, annot=annot, fmt=".2f", cmap=cmap, 
                   square=True, cbar_kws={"shrink": .8})
        plt.title(title)
        plt.tight_layout()
        plt.show()
        
        self['last_plot'] = {'title': title, 'cmap': cmap, 'figsize': figsize}
    
    def get_high_correlations(self, threshold: float = 0.7, exclude_self: bool = True) -> pd.DataFrame:
        """Get pairs of variables with high correlation."""
        if self.correlation_matrix is None:
            self.compute_analysis()
            
        corr_pairs = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i if exclude_self else 0, len(self.correlation_matrix.columns)):
                if exclude_self and i == j:
                    continue
                corr_value = self.correlation_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    corr_pairs.append({
                        'Variable_1': self.correlation_matrix.columns[i],
                        'Variable_2': self.correlation_matrix.columns[j],
                        'Correlation': corr_value
                    })
        
        high_corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
        self['high_correlations'] = high_corr_df
        return high_corr_df
    
    def plot_correlation_distribution(self, bins: int = 30) -> None:
        """Plot the distribution of correlation values."""
        if self.correlation_matrix is None:
            self.compute_analysis()
            
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool), k=1)
        correlations = self.correlation_matrix.where(mask).stack().values
        
        plt.figure(figsize=(8, 6))
        plt.hist(correlations, bins=bins, alpha=0.7, edgecolor='black')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Frequency')
        plt.title('Distribution of Correlation Coefficients')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        self['correlation_distribution'] = correlations


# Advanced usage examples:
if __name__ == "__main__":
    # Create sample data
    data = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.randn(100),
        'D': np.random.randn(100) * 0.8 + np.random.randn(100) * 0.2
    })
    
    print("=== Magic Methods Demo ===")
    
    # 1. Callable analyzer - auto execute with plot
    print("\n1. Callable execution:")
    analyzer = CorrelationAnalyzer(data)()
    
    # 2. Context manager usage
    print("\n2. Context manager:")
    with CorrelationAnalyzer(data) as ca:
        ca.get_high_correlations(threshold=0.1)
    # Auto-executes analysis on exit
    
    # 3. Dictionary-like access
    print("\n3. Dictionary-like access:")
    analyzer = CorrelationAnalyzer(data)
    analyzer.compute_analysis()
    print(f"Correlation matrix shape: {analyzer['correlation_matrix'].shape}")
    print(f"Method used: {analyzer['method']}")
    
    # 4. String representations
    print("\n4. String representations:")
    print(f"Repr: {repr(analyzer)}")
    print(f"Str:\n{str(analyzer)}")
    
    # 5. Boolean and length operations
    print(f"\n5. Boolean/Length: Has results: {bool(analyzer)}, Count: {len(analyzer)}")
    
    # 6. Iteration over results
    print(f"\n6. Available results: {list(analyzer)}")
    
    # 7. Membership testing
    print(f"\n7. 'correlation_matrix' in analyzer: {'correlation_matrix' in analyzer}")
    
    # 8. Combining analyzers
    print("\n8. Combining analyzers:")
    analyzer2 = CorrelationAnalyzer(data)
    analyzer2.compute_analysis(method='spearman')
    combined = analyzer + analyzer2
    print(f"Combined keys: {list(combined.keys())}")
    
    # 9. No-cache context
    print("\n9. No-cache execution:")
    with analyzer.no_cache():
        result = analyzer.compute_analysis()
    
    # 10. Results property
    print(f"\n10. All results: {list(analyzer.results.keys())}")