from typing import List
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from core.feature_selection.base_feature_selector import BaseFeatureSelector


class RandomForestFeatureSelector(BaseFeatureSelector):
    def __init__(self, X_train=None, y_train=None, max_features=None, threshold='mean'):
        super().__init__(X_train=X_train, y_train=y_train)
        self.max_features = max_features
        self.threshold = threshold
        self.selected_features_mask_ = None
        self.feature_names_ = None

    def _create_selector(self):
        """Create the actual selector implementation."""
        estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        selector = SelectFromModel(
            estimator=estimator,
            max_features=self.max_features,
            threshold=self.threshold
        )
        return selector

    def get_search_space(self) -> dict:
        """Define the hyperparameter search space."""
        n_features = 100 if self.X_train is None else self.X_train.shape[1]
        return {
            'feature_selection__max_features': list(range(1, n_features + 1)),
            'feature_selection__threshold': ['mean', 'median']
        }

    def fit(self, X, y):
        # Garantir que X seja DataFrame e preserve nomes das features
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_ = list(X.columns)

        if self.selector is None:
            self.selector = self._create_selector()

        self.selector.fit(X, y)

        # Armazenar máscara de features selecionadas
        self.selected_features_mask_ = self.selector.get_support()

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        transformed = self.selector.transform(X.values)
        # Retorna DataFrame com nomes das features selecionadas
        selected_features = self._get_selected_features()
        return pd.DataFrame(transformed, columns=selected_features, index=X.index)

    def _get_selected_features(self) -> List[str]:
        """Retorna nomes das features selecionadas"""
        if not hasattr(self, 'selected_features_mask_') or self.selected_features_mask_ is None:
            return self.feature_names_

        return [name for name, selected in zip(self.feature_names_, self.selected_features_mask_)
                if selected]
