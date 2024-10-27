from sklearn.decomposition import PCA

from core.feature_selection.base_feature_selector import BaseFeatureSelector

class PCAFeatureSelector(BaseFeatureSelector):
    def __init__(self, X_train, n_components=0.95, step=50):
        self.step = step
        self.n_components = n_components
        self.max_components = X_train.shape[1]
        super().__init__(X_train)

    def _create_selector(self, n_components=0.95):
        selector = PCA(n_components=min(n_components, self.max_components))
        return selector

    def get_search_space(self):
        if isinstance(self.n_components, float):
            # Para valores percentuais, retornar lista de valores
            return {'feature_selection__n_components': [0.7, 0.8, 0.85, 0.9, 0.95, 0.99]}
        else:
            # Para valores inteiros, retornar lista de valores
            max_components = min(self.max_components, 
                               self.n_components if isinstance(self.n_components, int) 
                               else self.max_components)
            return {'feature_selection__n_components': list(range(10, max_components + 1, self.step))}
