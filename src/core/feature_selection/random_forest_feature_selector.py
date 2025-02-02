from typing import List
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from core.feature_selection.base_feature_selector import BaseFeatureSelector

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RandomForestFeatureSelector(BaseFeatureSelector):
    def __init__(self, X_train=None, y_train=None, max_features=None, threshold='mean'):
        super().__init__(X_train=X_train, y_train=y_train)

        self.max_features = max_features
        self.threshold = threshold


    def _create_selector(self):
        estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        if isinstance(self.threshold, str):
            # Mapa de funções para cálculo do threshold
            threshold_map = {
                'mean': 'mean',
                'median': 'median',
                '0.5*mean': lambda x: 0.5 * np.mean(x),
                '1.5*mean': lambda x: 1.5 * np.mean(x)
            }
            threshold = threshold_map.get(self.threshold, 'mean')
        else:
            threshold = self.threshold

        selector = SelectFromModel(
            estimator=estimator,
            max_features=self.max_features,
            threshold=threshold  
        )

        return selector

    def _get_selected_features(self) -> List[str]:
        """
        Retorna as features selecionadas usando a máscara armazenada.
        """
        if self.selected_features_mask_ is None:
            return self.feature_names_  # Retorna todas as features se ainda não foi feito fit

        return [name for name, selected in zip(self.feature_names_, self.selected_features_mask_) if selected]

    def get_search_space(self):
        if self.X_train is None:
            return {
                'feature_selection__max_features': list(range(1, 101)),
                'feature_selection__threshold': ['mean', 'median', '0.5*mean', '1.5*mean']
            }

        n_features = self.X_train.shape[1]
        return {
            'feature_selection__max_features': list(range(1, n_features + 1)),
            'feature_selection__threshold': ['mean', 'median', '0.5*mean', '1.5*mean']
        }

    def get_params(self, deep=True):
        return {
            'max_features': self.max_features,
            'threshold': self.threshold
        }

    def set_params(self, **params):
        """Método set_params melhorado com verificação segura de selector"""
        # Primeiro atualizamos os parâmetros
        any_param_updated = False
        for param, value in params.items():
            if param in ['max_features', 'threshold']:  # Apenas parâmetros válidos
                setattr(self, param, value)
                any_param_updated = True

        # Verificação segura de selector usando hasattr
        if any_param_updated and hasattr(self, 'selector') and self.selector is not None:
            # Recriamos o selector e fazemos fit se tivermos dados
            self.selector = self._create_selector()
            if hasattr(self, 'X_train') and hasattr(self, 'y_train') and \
               self.X_train is not None and self.y_train is not None:
                self.selector.fit(self.X_train, self.y_train)

        return self

    def fit(self, X, y):
        """Adicionado para atualizar o threshold após o fit"""
        if X is None:
            raise ValueError("X cannot be None")

        super().fit(X, y)

        # Após o fit, podemos atualizar o threshold se necessário
        if not isinstance(self.threshold, (int, float)):
            if not hasattr(self.selector, 'estimator_'):
                raise ValueError("Selector's estimator not fitted properly")

            importances = self.selector.estimator_.feature_importances_
            threshold_map = {
                'mean': lambda x: np.mean(x),
                'median': lambda x: np.median(x),
                '0.5*mean': lambda x: 0.5 * np.mean(x),
                '1.5*mean': lambda x: 1.5 * np.mean(x)
            }

            self.selected_features_mask_ = self.selector.get_support()

        return self
