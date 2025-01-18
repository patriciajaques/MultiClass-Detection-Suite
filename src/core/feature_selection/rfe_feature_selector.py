from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

from core.feature_selection.base_feature_selector import BaseFeatureSelector

class RFEFeatureSelector(BaseFeatureSelector):
    def __init__(self, X_train, y_train, n_features_to_select=10):
        super().__init__(X_train, y_train, n_features_to_select=n_features_to_select)

    def _create_selector(self, n_features_to_select=10):
        estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        
        selector = RFECV(
            estimator=estimator,
            step=0.1,             # Remove 10% das features por vez
            min_features_to_select=10,  # Mínimo de features a manter
            cv=5,                 # 5-fold CV para ser mais rápido
            scoring='balanced_accuracy',  # Métrica adequada para classes desbalanceadas
            n_jobs=-1            # Paralelização
        )
        return selector

    def get_search_space(cls):
        # Com RFECV, não precisamos especificar n_features_to_select
        # pois ele encontra automaticamente o número ótimo
        return {}