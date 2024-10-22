from sklearn.decomposition import PCA

from core.feature_selection.base_feature_selector import BaseFeatureSelector

class PCAFeatureSelector(BaseFeatureSelector):
    def __init__(self, X_train, n_components=0.95, step=50):
        super().__init__(X_train, n_components=n_components)
        self.step = step
        self.n_components = n_components

    def _create_selector(self, n_components=0.95):
        n_features = self.X_train.shape[1]
        selector = PCA(n_components=min(n_components, n_features))
        return selector

    def get_search_space(self):
        return {'feature_selection__n_components': list(range(10, self.n_components + 1, self.step))}