from skopt import BayesSearchCV
from model_training import ModelTraining
from model_params import get_bayes_search_spaces
from feature_selection import FeatureSelection
from logger_config import LoggerConfig
import logging

class BayesianOptimizationTraining(ModelTraining):
    """
    Subclasse específica para treinamento usando otimização bayesiana.
    """
    def __init__(self):
        # Configuração do logger no construtor
        super().__init__() 
        LoggerConfig.configure_log_file('bayesian_optimization', '.log')

    def optimize_model(self, pipeline, model_name, selector_name, X_train, y_train, n_iter, cv, scoring):

        logging.info(f"Training and evaluating {model_name} with BayesianOptimization and {selector_name}:")
        print(f"Training and evaluating {model_name} with BayesianOptimization and {selector_name}:")

        # Bayesian Optimization
        search_space = get_bayes_search_spaces()[model_name]
        logging.info(f"Running Bayesian optimization for {model_name} with selector {selector_name}")
        logging.info(f"Search space: {search_space}")
        # Adicionar parâmetros para o seletor ao espaço de busca
        search_space.update(FeatureSelection.get_search_spaces().get(selector_name, {}))
        # Best model será um objeto do tipo Pipeline
        best_model, best_result, opt = self.execute_bayesian_optimization(pipeline, search_space, X_train, y_train, n_iter=n_iter, cv=cv, scoring=scoring)
        # Extraindo as predições de validação cruzada diretamente de cv_results_
        logging.info(f"Bayesian optimization results: Melhor resultado: {best_result}, Resultado médio da validação cruzada: {opt.cv_results_['mean_test_score'][opt.best_index_]}")

        # Armazenando mais informações sobre a configuração
        self.trained_models[f"{model_name}_{selector_name}"] = {
            'model': best_model,
            'training_type': "Bayesian Optimization",
            'hyperparameters': best_model.get_params(),  # Pegando hiperparâmetros do modelo
            'cv_result': best_result
        }
        logging.info(f"BayesianOptimization Best Result for {model_name} with {selector_name}: {best_result}")
        print(f"BayesianOptimization Best Result for {model_name} with {selector_name}: {best_result}")


    def execute_bayesian_optimization(self, pipeline, space, X_train, y_train, n_iter=50, cv=5, scoring='balanced_accuracy'):
        search = BayesSearchCV(
            pipeline,
            search_spaces=space,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=2,
            random_state=42,
            #return_train_score=True,
            verbose=3
        )

        search.fit(X_train, y_train, callback=LoggerConfig.log_results)
        best_model = search.best_estimator_

        return best_model, search.best_score_, search


