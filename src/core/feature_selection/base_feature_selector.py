from sklearn.base import BaseEstimator, TransformerMixin

class BaseFeatureSelector(BaseEstimator, TransformerMixin):
    """Base class for feature selectors with sklearn compatibility."""

    def __init__(self, X_train=None, y_train=None, **kwargs):
        self.X_train = X_train
        self.y_train = y_train
        self.selector = None
        super().__init__()
        self.set_params(**kwargs)

    def fit(self, X, y=None):
        """Fit the selector to data."""
        self.selector = self._create_selector()
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        """Transform the data."""
        if self.selector is None:
            raise ValueError("Selector not fitted. Call fit first.")
        return self.selector.transform(X)

    def _create_selector(self, **kwargs):
        """Abstract method to create the actual selector."""
        raise NotImplementedError

    def get_params(self, deep=True):
        """Get parameters. Required for sklearn compatibility."""
        return self.__dict__

    def set_params(self, **params):
        """Set parameters. Required for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
