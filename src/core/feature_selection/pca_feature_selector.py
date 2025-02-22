"""
Copyright (c) 2025 Patricia Jaques
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License. To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
"""

from typing import List
import numpy as np
from sklearn.decomposition import PCA
from core.feature_selection.base_feature_selector import BaseFeatureSelector


class PCAFeatureSelector(BaseFeatureSelector):
    """
    Implementa seleção de características usando PCA com compatibilidade scikit-learn.
    Só precisa sobrescrever _create_selector e _get_selected_features,
    pois o BaseFeatureSelector já lida com fit/transform/genéricos.
    """

    def __init__(self, n_components=0.95):
        """
        Args:
            n_components: Número de componentes ou proporção da variância (0 < n_components <= 1)
        """
        super().__init__()
        self.n_components = n_components
        self.n_features_ = None  # Número de colunas de X (definido no fit)

    def fit(self, X, y=None):
        """
        Precisamos sobrescrever o fit apenas para descobrir X.shape[1]
        antes de chamarmos o fit genérico do BaseFeatureSelector.
        """
        self.n_features_ = X.shape[1]
        return super().fit(X, y)  # chamará _create_selector() e selector.fit(X, y)

    def _create_selector(self):
        """
        Cria e retorna o objeto PCA, ajustando n_components
        se for um int maior que o número real de colunas.
        """
        components = self.n_components
        if isinstance(components, int) and self.n_features_ is not None:
            components = min(components, self.n_features_)
        return PCA(n_components=components)

    def _get_selected_features(self) -> List[str]:
        """
        Em PCA não existe 1-para-1 com as features originais,
        então retornamos nomes genéricos 'PC1', 'PC2', etc.
        """
        if not self._is_fitted:
            raise ValueError("PCA não foi ajustado. Execute fit primeiro.")
        n_components = self.selector.n_components_
        return [f"PC{i+1}" for i in range(n_components)]

    def get_search_space(self):
        """
        Retorna o espaço de busca para otimização.
        """
        if isinstance(self.n_components, float):
            return {
                'feature_selection__n_components': {
                    'type': 'float',
                    'range': [0.5, 0.99],
                    'values': [0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
                }
            }
        else:
            max_features = self.n_features_ if self.n_features_ is not None else 64
            return {
                'feature_selection__n_components': {
                    'type': 'int',
                    'range': [1, max_features]
                }
            }
