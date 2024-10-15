from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from core.training.model_params import get_models
from core.preprocessors.feature_selection import FeatureSelection

class ModelTraining(ABC):
    def __init__(self):
        self.trained_models = {}

    def train_model(self, X_train, y_train, n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1):
        selectors = FeatureSelection.create_selectors(X_train, y_train)
        models = get_models()

        for model_name, model_config in models.items():
            for selector_name, selector in selectors.items():
                pipeline = self._create_pipeline(selector, model_config)
                self.optimize_model(pipeline, model_name, selector_name, X_train, y_train, n_iter, cv, scoring, n_jobs)

        return self.trained_models

    @abstractmethod
    def optimize_model(self, pipeline, model_name, selector_name, X_train, y_train, n_iter, cv, scoring, n_jobs):
        pass

    @staticmethod
    def _create_pipeline(selector, model_config):
        return Pipeline([
            ('feature_selection', selector),
            ('classifier', model_config)
        ])