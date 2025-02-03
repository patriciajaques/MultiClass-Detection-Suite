from typing import List
from sklearn.decomposition import PCA
from core.feature_selection.base_feature_selector import BaseFeatureSelector
import numpy as np


class PCAFeatureSelector(BaseFeatureSelector):
    """
    Implementa seleção de características usando PCA com compatibilidade total scikit-learn.
    """

    def __init__(self, n_components=0.95):
        """
        Inicializa o seletor PCA.
        
        Args:
            n_components: Número de componentes ou proporção da variância (0 < n_components <= 1)
        """
        self.n_components = n_components
        self.n_features_ = None  # Armazena número de features após fit
        super().__init__()

    def fit(self, X, y=None):
        """
        Ajusta o PCA aos dados e armazena informações importantes.
        
        Args:
            X: Dados de treino
            y: Target (ignorado para PCA)
        """
        self.n_features_ = X.shape[1]
        self.selector = self._create_selector()
        self.selector.fit(X)
        return self

    def transform(self, X):
        """
        Transforma os dados usando o PCA ajustado.
        
        Args:
            X: Dados para transformar
        """
        if self.selector is None:
            raise ValueError("PCA não foi ajustado. Execute fit primeiro.")
        return self.selector.transform(X)

    def _create_selector(self):
        """
        Cria o seletor PCA com os parâmetros atuais.
        """
        components = self.n_components
        if isinstance(components, int) and self.n_features_ is not None:
            components = min(components, self.n_features_)

        return PCA(n_components=components)

    def get_search_space(self):
        """
        Retorna o espaço de busca para otimização de hiperparâmetros.
        Usa valores padrão seguros quando n_features_ não está disponível.
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
            # Usa valor padrão seguro quando n_features_ não está disponível
            max_features = self.n_features_ if self.n_features_ is not None else 64
            return {
                'feature_selection__n_components': {
                    'type': 'int',
                    'range': [1, max_features]
                }
            }

    def get_feature_names_out(self, input_features=None):
        """
        Retorna os nomes das features transformadas.
        
        Args:
            input_features: Nomes das features originais (ignorado para PCA)
        """
        if self.selector is None:
            raise ValueError("PCA não foi ajustado. Execute fit primeiro.")
        n_components = self.selector.n_components_
        return [f'PC{i+1}' for i in range(n_components)]

    def get_params(self, deep=True):
        """Implementa get_params para compatibilidade com scikit-learn."""
        return {
            'n_components': self.n_components
        }

    def set_params(self, **params):
        """Implementa set_params para compatibilidade com scikit-learn."""
        for param, value in params.items():
            setattr(self, param, value)
        return self
    
    def _get_selected_features(self) -> List[str]:
        """
        Para PCA, os "nomes" são PC1, PC2, etc.
        """
        if self.selector is None:
            raise ValueError("PCA não foi ajustado. Execute fit primeiro.")
        # n_components_ é o número final de componentes do PCA
        n_components = self.selector.n_components_
        return [f"PC{i+1}" for i in range(n_components)]
