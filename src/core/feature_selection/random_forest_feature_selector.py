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

    def _create_selector(self):
        """Cria o seletor com configuração básica"""
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)

        selector = SelectFromModel(
            estimator=estimator,
            max_features=self.max_features,
            threshold=self.threshold  # Usa threshold direto
        )
        return selector

    def fit(self, X, y):
        """Fit com armazenamento da máscara de features"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.X_train = X
        self.y_train = y
        self.feature_names_ = list(X.columns)

        if self.selector is None:
            self.selector = self._create_selector()

        self.selector.fit(X, y)
        self.selected_features_mask_ = self.selector.get_support()
        return self

    def _get_selected_features(self) -> List[str]:
        """Retorna features selecionadas"""
        if self.selected_features_mask_ is None:
            return self.feature_names_
        return [name for name, selected in
                zip(self.feature_names_, self.selected_features_mask_)
                if selected]

    def get_search_space(self):
        """Espaço de busca para otimização"""
        n_features = 100 if self.X_train is None else self.X_train.shape[1]
        return {
            'feature_selection__max_features': list(range(1, n_features + 1)),
            'feature_selection__threshold': ['mean', 'median']
        }