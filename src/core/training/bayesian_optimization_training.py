from skopt import BayesSearchCV
from core.training.model_training import ModelTraining
from core.training.model_params import get_bayes_search_spaces
from core.preprocessors.feature_selection import FeatureSelection
from core.logging.logger_config import LoggerConfig
import logging

class BayesianOptimizationTraining(ModelTraining):
    def __init__(self):
        super().__init__()
        LoggerConfig.configure_log_file('bayesian_optimization', '.log')

    def optimize_model(self, pipeline, model_name, selector_name, X_train, y_train, n_iter, cv, scoring, n_jobs=-1):
        logging.info(f"Training and evaluating {model_name} with BayesianOptimization and {selector_name}")
        print(f"Training and evaluating {model_name} with BayesianOptimization and {selector_name}")

        search_space = self._prepare_search_space(model_name, selector_name)
        best_model, best_result, opt = self._execute_bayesian_optimization(pipeline, search_space, X_train, y_train, n_iter, cv, scoring, n_jobs)

        logging.info(f"Bayesian optimization results: Best result: {best_result}, Average cross-validation result: {opt.cv_results_['mean_test_score'][opt.best_index_]}")

        self._store_model_results(model_name, selector_name, best_model, best_result)

    def _prepare_search_space(self, model_name, selector_name):
        search_space = get_bayes_search_spaces()[model_name]
        selector_search_space = FeatureSelection.get_search_spaces().get(selector_name, {})

        if isinstance(search_space, list):
            for subspace in search_space:
                subspace.update(selector_search_space)
        else:
            search_space.update(selector_search_space)

        logging.info(f"Search space for {model_name} with selector {selector_name}: {search_space}")
        return search_space

    def _execute_bayesian_optimization(self, pipeline, space, X_train, y_train, n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1):
        search = BayesSearchCV(
            pipeline,
            search_spaces=space,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=42,
            verbose=3
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
        logging.info(f"BayesianOptimization Best Result for {model_name} with {selector_name}: {best_result}")
        print(f"BayesianOptimization Best Result for {model_name} with {selector_name}: {best_result}")