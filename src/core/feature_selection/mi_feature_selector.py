from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from core.feature_selection.base_feature_selector import BaseFeatureSelector

class MutualInformationFeatureSelector(BaseFeatureSelector):
    def __init__(self, X_train=None, y_train=None, k=10):
        super().__init__(X_train=X_train, y_train=y_train, k=k)

    def _create_selector(self, k=10):
        selector = SelectKBest(mutual_info_classif, k=k)
        selector.fit(self.X_train, self.y_train)
        return selector

    def get_search_space(self):
        return {'feature_selection__k': [5, 10, 20, 30, 40, 50, 'all']}