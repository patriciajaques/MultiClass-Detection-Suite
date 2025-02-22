"""
Copyright (c) 2025 Patricia Jaques
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License. To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
"""

from core.feature_selection.mi_feature_selector import MutualInformationFeatureSelector
from core.feature_selection.pca_feature_selector import PCAFeatureSelector
from core.feature_selection.random_forest_feature_selector import RandomForestFeatureSelector
from core.feature_selection.rfe_feature_selector import RFEFeatureSelector


class FeatureSelectionFactory:
    SELECTORS = {
        'rfe': RFEFeatureSelector,
        'pca': PCAFeatureSelector,
        'rf': RandomForestFeatureSelector,
        'mi': MutualInformationFeatureSelector,
        'none': None
    }

    @staticmethod
    def create_selector(method: str, **kwargs):
        """
        Cria e retorna o seletor de features escolhido.
        
        Args:
            method (str): Nome do seletor ('rfe', 'pca', 'rf', 'mi', 'none').
            **kwargs: Hiperparâmetros específicos para o construtor do seletor.
                      (Por exemplo, k=10 para MutualInformation, 
                       n_components=0.95 para PCA, etc.)
        Returns:
            Um objeto que herda de BaseFeatureSelector ou None se 'none'
        """
        if method not in FeatureSelectionFactory.SELECTORS:
            raise ValueError(f"Método desconhecido: {method}")

        selector_class = FeatureSelectionFactory.SELECTORS[method]
        if selector_class is None:
            return None

        # Agora, sem X_train ou y_train:
        # passamos somente hyperparams pro construtor
        return selector_class(**kwargs)
