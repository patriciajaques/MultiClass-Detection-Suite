import numpy as np

from core.feature_selection.pca_feature_selector import PCAFeatureSelector
from core.feature_selection.random_forest_feature_selector import RandomForestFeatureSelector
from core.feature_selection.rfe_feature_selector import RFEFeatureSelector
from core.feature_selection.mi_feature_selector import MutualInformationFeatureSelector

class FeatureSelectionFactory:
    SELECTORS = {
        'rfe': RFEFeatureSelector,
        'pca': PCAFeatureSelector,
        'rf': RandomForestFeatureSelector,
        'mi': MutualInformationFeatureSelector,
        'none': None
    }

    @staticmethod
    def create_selector(method, X_train, y_train=None, **kwargs):
        if method not in FeatureSelectionFactory.SELECTORS:
            raise ValueError(f"Método desconhecido: {method}")
        
        selector_class = FeatureSelectionFactory.SELECTORS[method]
        if method in ['rfe', 'mi', 'rf']:
            selector = selector_class(X_train, y_train, **kwargs)
        else:
            selector = selector_class(X_train, **kwargs)
        
        return selector

    @staticmethod
    def get_available_selectors_names():
        """
        Retorna uma lista dos métodos de seleção de características disponíveis.
        """
        return list(FeatureSelectionFactory.SELECTORS.keys())

    @staticmethod
    def extract_selected_features(pipeline, feature_names):
        """
        Extrai as características selecionadas pelo seletor de características no pipeline.

        Args:
            pipeline: Pipeline treinado.
            feature_names: Lista de nomes das características originais.

        Returns:
            List: Lista de características selecionadas.
        """
        if 'feature_selection' not in pipeline.named_steps:
            return feature_names
        
        selector = pipeline.named_steps['feature_selection']

        if hasattr(selector, 'get_support'):
            mask = selector.get_support()
            selected_features = np.array(feature_names)[mask]
        elif hasattr(selector, 'transform'):
            # Para métodos como PCA que transformam as características
            transformed = selector.transform(np.identity(len(feature_names)))
            # Retornar os componentes principais como nomes
            selected_features = [f'PC{i+1}' for i in range(transformed.shape[1])]
        else:
            raise ValueError("O seletor não tem métodos para extrair características.")

        return selected_features
