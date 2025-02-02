from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, List


class BaseFeatureSelector(BaseEstimator, TransformerMixin, ABC):
    """Abstract base class for feature selectors with sklearn compatibility."""

    def __init__(self, X_train=None, y_train=None, **kwargs):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names_ = None
        self.selector = None

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
        self.X_train = X
        self.y_train = y

        # Guardar nomes das features
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]

        # Inicializa o selector se necessário
        if self.selector is None:
            self.selector = self._create_selector()

        self.selector.fit(X, y)
        return self

    def transform(self, X):
        """Transform the data with validation."""
        if self.selector is None:
            raise ValueError("Selector not fitted. Call fit first.")
        return self.selector.transform(X)

    @abstractmethod
    def _get_selected_features(self) -> List[str]:
        """
        Método abstrato que deve ser implementado por cada seletor 
        para retornar suas features selecionadas.
        
        Returns:
            List[str]: Lista com nomes das features selecionadas
        """
        pass

    def get_feature_names(self) -> List[str]:
        """
        Interface pública para obter nomes das features selecionadas.
        
        Returns:
            List[str]: Lista com nomes das features selecionadas
        
        Raises:
            ValueError: Se o seletor não foi fitted
        """
        if self.selector is None:
            raise ValueError("Selector not fitted. Call fit first.")

        if self.feature_names_ is None:
            return [f'feature_{i}' for i in range(self.X_train.shape[1])]

        return self._get_selected_features()
