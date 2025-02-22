"""
Copyright (c) 2025 Patricia Jaques
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License. To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
"""

from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, List


class BaseFeatureSelector(BaseEstimator, TransformerMixin, ABC):
    """Abstract base class for feature selectors with sklearn compatibility."""

    def __init__(self, **kwargs):
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
    
    @abstractmethod
    def _get_selected_features(self) -> List[str]:
        """
        Método abstrato que deve ser implementado por cada seletor 
        para retornar suas features selecionadas.
        
        Returns:
            List[str]: Lista com nomes das features selecionadas
        """
        pass

    def fit(self, X, y=None):
        """Armazena feature_names_ e chama self.selector.fit(X, y)."""
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = None

        if self.selector is None:
            self.selector = self._create_selector()

        self.selector.fit(X, y)
        self._is_fitted = True
        return self

    def transform(self, X):
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Use fit first.")
        return self.selector.transform(X)
    
    def get_support(self):
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Use fit first.")
        return self.selector.get_support()

    def _more_tags(self):
        # se quiser dizer que esse selector é "requires_y"
        return {'requires_y': True}
