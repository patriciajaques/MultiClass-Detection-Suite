# bayesian_optimization_training.py

from skopt import BayesSearchCV
from core.training.model_training import ModelTraining
from core.training.skopt_model_params import SkoptModelParams
from core.logging.logger_config import LoggerConfig, with_logging
import logging

@with_logging('skopt_training')
class SkoptBayesianOptimizationTraining(ModelTraining):
    def __init__(self):
        super().__init__()

    def optimize_model(self, pipeline, model_name, selector_name, X_train, y_train, n_iter, cv, scoring, n_jobs=-1, selector_search_space=None):
        self.logger.info(f"Training and evaluating {model_name} with Bayesian Optimization and {selector_name}")
        print(f"Training and evaluating {model_name} with Bayesian Optimization and {selector_name}")

        # Obter o espaço de busca específico do modelo
        search_space_model = SkoptModelParams.get_bayes_search_spaces().get(model_name, {})
        # Obter o espaço de busca específico do seletor (já passado como parâmetro)
        search_space_selector = selector_search_space

        # Combinar os espaços de busca do modelo e do seletor
        if isinstance(search_space_model, list):
            for subspace in search_space_model:
                subspace.update(search_space_selector)
        else:
            search_space_model.update(search_space_selector)

        self.logger.info(f"Search space for {model_name} with selector {selector_name}: {search_space_model}")

        best_model, best_result, opt = self._execute_bayesian_optimization(
            pipeline,
            search_space_model,
            X_train,
            y_train,
            n_iter,
            cv,
            scoring,
            n_jobs
        )

        # Log the results using ModelTraining's method
        self.log_search_results(self.logger, opt, model_name, selector_name)

        logging.info(f"Bayesian optimization results: Best result: {best_result}, "
                     f"Average cross-validation result: {opt.cv_results_['mean_test_score'][opt.best_index_]}")
        
        self._store_model_results(model_name, selector_name, best_model, best_result)

    def _execute_bayesian_optimization(self, pipeline, space, X_train, y_train, n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1):
        search = BayesSearchCV(
            pipeline,
            search_spaces=space,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=42,
            verbose=2
        )

        search.fit(X_train, y_train, callback=LoggerConfig.log_results)
        return search.best_estimator_, search.best_score_, search

    def _store_model_results(self, model_name, selector_name, best_model, best_result):
        self.trained_models[f"{model_name}_{selector_name}"] = {
            'model': best_model,
            'training_type': "Bayesian Optimization",
            'hyperparameters': best_model.get_params(),
            'cv_result': best_result
        }
        self.logger.info(f"Bayesian Optimization Best Result for {model_name} with {selector_name}: {best_result}")
        print(f"Bayesian Optimization Best Result for {model_name} with {selector_name}: {best_result}")
