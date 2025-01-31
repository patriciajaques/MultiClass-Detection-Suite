from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Any


class BaseFeatureSelector(BaseEstimator, TransformerMixin, ABC):
    """Abstract base class for feature selectors with sklearn compatibility."""

    def __init__(self, X_train=None, y_train=None, **kwargs):
        self.X_train = X_train
        self.y_train = y_train
        self.selector = None
        # self.set_params(**kwargs)


    @abstractmethod
    def _create_selector(self, **kwargs) -> Any:
        """Create the actual selector implementation."""
        pass

    @abstractmethod
    def get_search_space(self) -> dict:
        """Define the hyperparameter search space."""
        pass

    def fit(self, X, y=None):
        """Fit the selector to data with validation."""
        if self.selector is None:
            self.X_train = X
            self.y_train = y
            self.selector = self._create_selector()

        self.selector.fit(X, y)
        return self


    def transform(self, X):
        """Transform the data with validation."""
        if self.selector is None:
            raise ValueError("Selector not fitted. Call fit first.")
        return self.selector.transform(X)
