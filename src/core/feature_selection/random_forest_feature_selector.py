import numpy as np
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from core.feature_selection.base_feature_selector import BaseFeatureSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomForestFeatureSelector(BaseFeatureSelector):
    def __init__(self, X_train, y_train, max_features=None):
        super().__init__(X_train, y_train)
        self.max_features = max_features
        self.selector = self._create_selector() if X_train is not None else None

    # Em random_forest_feature_selector.py, método _create_selector:


    def _create_selector(self):

        # 1. Cria o estimador Random Forest base
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)

        # 2. Verifica se os dados de entrada são um DataFrame
        if isinstance(self.X_train, pd.DataFrame):
            # 2.1 Se for DataFrame, guarda os nomes das colunas
            feature_names = self.X_train.columns.tolist()

            # 2.2 Treina o Random Forest
            estimator.fit(self.X_train, self.y_train)

            # 2.3 Cria o seletor de features usando o modelo já treinado
            selector = SelectFromModel(
                estimator,
                max_features=self.max_features,
                prefit=True  # Indica que o estimador já está treinado
            )

            # 2.4 Armazena explicitamente os nomes das features no seletor
            selector.feature_names_in_ = feature_names
        else:
            # 3. Se não for DataFrame, segue o fluxo normal
            estimator.fit(self.X_train, self.y_train)
            selector = SelectFromModel(
                estimator,
                max_features=self.max_features
            )

        # 4. Log do número de features selecionadas
        selected_features = selector.get_support().sum()
        logger.info(f"Selected {selected_features} features")

        return selector

    def get_search_space(self):
        n_features = self.X_train.shape[1]
        max_features_range = list(range(1, n_features + 1))
        return {
            'feature_selection__max_features': max_features_range,
            'feature_selection__threshold': ['mean', 'median', '0.5*mean', '1.5*mean']
        }

    def set_params(self, **params):
        if 'max_features' in params:
            self.max_features = int(params['max_features'])
            self.selector.max_features = self.max_features
        
        if 'threshold' in params:
            if isinstance(params['threshold'], str):
                estimator = self.selector.estimator
                feature_importances = estimator.feature_importances_
                if params['threshold'] == 'mean':
                    threshold = np.mean(feature_importances)
                elif params['threshold'] == 'median':
                    threshold = np.median(feature_importances)
                elif params['threshold'] == '0.5*mean':
                    threshold = 0.5 * np.mean(feature_importances)
                elif params['threshold'] == '1.5*mean':
                    threshold = 1.5 * np.mean(feature_importances)
                self.selector.threshold = threshold
            else:
                self.selector.threshold = params['threshold']
        return self