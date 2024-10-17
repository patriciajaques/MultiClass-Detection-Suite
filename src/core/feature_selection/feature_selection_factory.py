import numpy as np

from core.feature_selection.pca_feature_selector import PCAFeatureSelector
from core.feature_selection.random_forest_feature_selector import RandomForestFeatureSelector
from core.feature_selection.rfe_feature_selector import RFEFeatureSelector
from core.feature_selection.mi_feature_selector import MutualInformationFeatureSelector

class FeatureSelectionFactory:
    @staticmethod
    def create_selector(method, X_train, y_train=None, **kwargs):
        if method == 'rfe':
            selector = RFEFeatureSelector(X_train, y_train, **kwargs)
        elif method == 'pca':
            selector = PCAFeatureSelector(X_train, **kwargs)
        elif method == 'rf':
            selector = RandomForestFeatureSelector(X_train, y_train)
        elif method == 'mi':
            selector = MutualInformationFeatureSelector(X_train, y_train, **kwargs)
        else:
            raise ValueError(f"Método desconhecido: {method}")
        return selector

    @staticmethod
    def get_available_selectors():
        """
        Retorna uma lista dos métodos de seleção de características disponíveis.
        """
        return ['rfe', 'pca', 'rf', 'mi']

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
