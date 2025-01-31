import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from core.feature_selection.base_feature_selector import BaseFeatureSelector

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RandomForestFeatureSelector(BaseFeatureSelector):
    """
    Implementa seleção de features usando Random Forest com melhor gerenciamento de estado.
    """

    def __init__(self, X_train=None, y_train=None, max_features=None, threshold='mean'):
        """
        Inicializa o seletor com parâmetros específicos.
        
        Args:
            X_train: Dados de treino
            y_train: Labels de treino
            max_features: Número máximo de features para selecionar
            threshold: Threshold para seleção ('mean', 'median', '0.5*mean', '1.5*mean' ou valor numérico)
        """
        self.max_features = max_features
        self.threshold = threshold
        self._is_fitted = False
        super().__init__(X_train=X_train, y_train=y_train)

    def _create_selector(self):
        """
        Cria e configura o seletor de features.
        """
        # Configura o estimador base
        estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        # Ajusta o número máximo de features se necessário
        if self.X_train is not None:
            n_features = self.X_train.shape[1]
            if self.max_features is None or self.max_features == 'auto':
                self.max_features = max(1, n_features // 2)
            elif isinstance(self.max_features, float):
                self.max_features = max(1, int(self.max_features * n_features))

        # Cria o seletor com os parâmetros atuais
        selector = SelectFromModel(
            estimator=estimator,
            max_features=self.max_features,
            threshold=self._get_threshold_value(self.threshold)
        )

        return selector

    def _get_threshold_value(self, threshold):
        """
        Calcula o valor do threshold baseado no tipo especificado.
        """
        if isinstance(threshold, (int, float)):
            return threshold

        if not hasattr(self.selector, 'estimator_'):
            return 'mean'

        importances = self.selector.estimator_.feature_importances_

        if threshold == 'mean':
            return np.mean(importances)
        elif threshold == 'median':
            return np.median(importances)
        elif threshold == '0.5*mean':
            return 0.5 * np.mean(importances)
        elif threshold == '1.5*mean':
            return 1.5 * np.mean(importances)
        else:
            return 'mean'

    def fit(self, X, y=None):
        """
        Ajusta o seletor aos dados, garantindo inicialização adequada.
        """
        self.X_train = X
        self.y_train = y

        if self.selector is None:
            self.selector = self._create_selector()

        self.selector.fit(X, y)
        self._is_fitted = True

        return self

    def transform(self, X):
        """
        Transforma os dados usando o seletor ajustado.
        """
        if not self._is_fitted:
            raise ValueError("Seletor não foi ajustado. Execute fit primeiro.")
        return self.selector.transform(X)

    def get_support(self):
        """
        Retorna máscara booleana das features selecionadas.
        """
        if not self._is_fitted:
            raise ValueError("Seletor não foi ajustado. Execute fit primeiro.")
        return self.selector.get_support()

    def get_search_space(self):
        """
        Define o espaço de busca para otimização de hiperparâmetros.
        """
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

    def set_params(self, **params):
        """
        Atualiza os parâmetros do seletor de forma segura.
        """
        for param, value in params.items():
            if param == 'max_features':
                self.max_features = value
            elif param == 'threshold':
                self.threshold = value

        # Se já estiver ajustado, recria o seletor com os novos parâmetros
        if self._is_fitted:
            self.selector = self._create_selector()
            self.selector.fit(self.X_train, self.y_train)

        return self

    def get_params(self, deep=True):
        """
        Retorna os parâmetros atuais do seletor.
        """
        return {
            'max_features': self.max_features,
            'threshold': self.threshold
        }
