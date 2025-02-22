"""
Copyright (c) 2025 Patricia Jaques
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License. To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
"""

# rfe_feature_selector.py

from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from core.feature_selection.base_feature_selector import BaseFeatureSelector


class RFEFeatureSelector(BaseFeatureSelector):
    def __init__(self, min_features_to_select=10, step=0.1):
        super().__init__()
        self.min_features_to_select = min_features_to_select
        self.step = step

    def _create_selector(self):
        base_estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1
        )
        return RFECV(
            estimator=base_estimator,
            step=self.step,
            min_features_to_select=self.min_features_to_select,
            cv=5,
            scoring='balanced_accuracy',
            n_jobs=1
        )

    def _get_selected_features(self) -> List[str]:
        if not self._is_fitted:
            raise ValueError("Seletor não foi ajustado. Use fit antes.")
        mask = self.selector.get_support()
        if self.feature_names_ is not None:
            return [name for name, m in zip(self.feature_names_, mask) if m]
        return [f"feature_{i}" for i, m in enumerate(mask) if m]

    def get_search_space(self) -> dict:
        return {
            'feature_selection__min_features_to_select': {
                'type': 'int',
                'range': [5, 50]
            },
            'feature_selection__step': {
                'type': 'float',
                'values': [0.1, 0.2, 0.3]
            }
        }
    

