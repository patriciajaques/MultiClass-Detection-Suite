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
        if selector_class is None:
            return None

        # Remover o parâmetro 'selector' se presente
        kwargs.pop('selector', None)

        try:
            if method == 'pca':
                return selector_class(X_train, **kwargs)
            else:
                return selector_class(X_train, y_train, **kwargs)
        except Exception as e:
            print(f"Error creating selector {method}: {str(e)}")
            raise

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
            pipeline: Pipeline treinado
            feature_names: Lista de nomes das características originais

        Returns:
            List[str]: Lista de características selecionadas
        """
        if 'feature_selection' not in pipeline.named_steps:
            return feature_names
        
        selector = pipeline.named_steps['feature_selection']
        
        try:
            # Agora usa o novo método get_feature_names()
            return selector.get_feature_names()
        except Exception as e:
            print(f"Erro ao extrair features: {str(e)}")
            return feature_names  # Fallback seguro
