"""
Copyright (c) 2025 Patricia Jaques
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License. To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
"""

# mutual_information_feature_selector.py

from typing import List
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from core.feature_selection.base_feature_selector import BaseFeatureSelector


class MutualInformationFeatureSelector(BaseFeatureSelector):
    """
    Seleciona `k` melhores features segundo mutual information.
    """

    def __init__(self, k=10):
        super().__init__()  # chama o construtor base
        self.k = k

    def _create_selector(self):
        # Aqui instanciamos um SelectKBest
        return SelectKBest(mutual_info_classif, k=self.k)

    def _get_selected_features(self) -> List[str]:
        if not self._is_fitted:
            raise ValueError("Seletor não foi ajustado. Execute fit primeiro.")
        mask = self.selector.get_support()
        if self.feature_names_ is not None:
            return [name for name, m in zip(self.feature_names_, mask) if m]
        return [f"feature_{i}" for i, m in enumerate(mask) if m]

    def get_search_space(self) -> dict:
        return {
            "feature_selection__k": [5, 10, 20, 30, 40, 50, 'all']
        }

