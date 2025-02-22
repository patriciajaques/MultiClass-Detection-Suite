"""
Copyright (c) 2025 Patricia Jaques
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License. To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
"""

from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from core.feature_selection.base_feature_selector import BaseFeatureSelector


class RandomForestFeatureSelector(BaseFeatureSelector):
    """
    Selecionador de features usando RandomForest+SelectFromModel.
    """

    def __init__(self, max_features=None, threshold='mean'):
        # Inicializa o BaseFeatureSelector (se tiver algo no construtor base)
        super().__init__()
        self.max_features = max_features
        self.threshold = threshold

    def _create_selector(self):
        """
        Cria o objeto SelectFromModel com um RandomForest base.
        Esse objeto é guardado em self.selector no fit() do BaseFeatureSelector.
        """
        estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        return SelectFromModel(
            estimator=estimator,
            max_features=self.max_features,
            threshold=self.threshold
        )

    def _get_selected_features(self) -> List[str]:
        """
        Implementa método abstrato do BaseFeatureSelector,
        retornando a lista de nomes das colunas selecionadas.
        """
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Use fit(...) first.")

        mask = self.selector.get_support()  # booleans
        # se a classe base armazenou .feature_names_ no fit, usamos:
        if self.feature_names_ is not None:
            return [name for name, selected in zip(self.feature_names_, mask) if selected]
        else:
            return [f"feature_{i}" for i, selected in enumerate(mask) if selected]

    def get_search_space(self) -> dict:
        """
        Retorna o espaço de busca de hiperparâmetros, se quiser otimizar.
        Se preferir algo fixo, pode deixar estático ou remover este método.
        """
        return {
            'feature_selection__max_features': [1, 5, 10, 20, 50, None],
            'feature_selection__threshold': ['mean', 'median', 0.1, 0.2]
        }
