from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline

from model_params import get_models
from feature_selection import FeatureSelection as fs

class ModelTraining(ABC):
    """
    Superclasse abstrata para treinamento de modelos.
    """

    def __init__(self):
        self.trained_models = {}  # Inicializa o dicionário para armazenar os resultados de cada modelo

    def train_model(self, X_train, y_train, n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1):

        selectors = fs.create_selectors(X_train, y_train)  # Criar seletores

        models = get_models()

        for model_name, model_config in models.items():
            for selector_name, selector in selectors.items():
                
                # Criar pipeline
                pipeline = self.create_pipeline(selector, model_config)
                
                self.optimize_model(pipeline, model_name, selector_name, X_train, y_train, n_iter, cv, scoring, n_jobs)
        return self.trained_models
        
    @abstractmethod
    def optimize_model(self, pipeline, model_name, selector_name, X_train, y_train, n_iter, cv, scoring):
        pass

    def create_pipeline(self, selector, model_config):
        # Cria o pipeline diretamente com o seletor e o modelo
        pipeline = Pipeline([
            ('feature_selection', selector),
            ('classifier', model_config)
        ])
        return pipeline
