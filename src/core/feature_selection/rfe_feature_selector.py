from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

from core.feature_selection.base_feature_selector import BaseFeatureSelector

class RFEFeatureSelector(BaseFeatureSelector):
    def __init__(self, X_train, y_train, n_features_to_select=10):
        super().__init__(X_train, y_train, n_features_to_select=n_features_to_select)

    def _create_selector(self, n_features_to_select=10):
        n_features = self.X_train.shape[1]
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=min(n_features_to_select, n_features))
        return selector

    def get_search_space(cls):
        return {'feature_selection__n_features_to_select': [1, 5, 10, 20, 30, 40, 50]}