from abc import ABC, abstractmethod
from .meta import AnalyzerMeta
from sklearn.linear_model import LinearRegression

class BaseAnalyzer(ABC, metaclass=AnalyzerMeta):

    @abstractmethod
    def analyze(self, df):
        pass

    @abstractmethod
    def visualize(self, df, **kwargs):
        pass

    def __call__(self, df):
        self.analyze(df)
        self.visualize(df)




class BaseFeatureSelector(ABC, metaclass=AnalyzerMeta):

    @abstractmethod
    def analyze(self, df):
        pass

    @abstractmethod
    def visualize(self, df, **kwargs):
        pass

    def __call__(self, df):
        self.analyze(df)
        self.visualize(df)
