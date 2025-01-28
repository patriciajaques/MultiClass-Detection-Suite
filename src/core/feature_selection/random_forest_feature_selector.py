import numpy as np
import logging
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

    def _create_selector(self):
        n_features = self.X_train.shape[1]
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        estimator.fit(self.X_train, self.y_train)
        
        # Se max_features não for especificado, use metade das features
        if self.max_features is None or self.max_features == 'auto':
            self.max_features = max(1, n_features // 2)
        elif isinstance(self.max_features, float):
            self.max_features = max(1, int(self.max_features * n_features))
        
        selector = SelectFromModel(estimator, max_features=self.max_features)
        
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