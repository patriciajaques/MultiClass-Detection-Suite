"""
Copyright (c) 2025 Patricia Jaques
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License. To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
"""

from sklearn.model_selection import RandomizedSearchCV
from core.logging.logger_config import LoggerConfig
from core.reporting.metrics_reporter import MetricsReporter
from core.training.base_training import BaseTraining
from core.models.parameter_handlers.grid_search_param_converter import GridSearchParamConverter


class RandomSearchTraining(BaseTraining):
    def __init__(self):
        super().__init__()
        self.logger = LoggerConfig.get_logger('random_search_training')

    def optimize_model(self, pipeline, model_name, model_params, selector_name, X_train, y_train, n_iter, cv, scoring, n_jobs=-1, selector_search_space=None, groups=None):
        try:
            self.logger.info(
                f"Training and evaluating {model_name} with RandomizedSearchCV and {selector_name}")

            param_grid = GridSearchParamConverter.convert_param_space(
                model_params, model_name)
            if selector_search_space:
                if isinstance(param_grid, list):
                    for subspace in param_grid:
                        subspace.update(selector_search_space)
                else:
                    param_grid.update(selector_search_space)

            random_search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                n_jobs=n_jobs,
                scoring=scoring,
                verbose=0,
                # Retorna -inf em vez de levantar erro
                error_score=float('-inf'),
                random_state=42
            )

            random_search.fit(X_train, y_train)

            # Exporta os resultados do CV utilizando o método específico
            MetricsReporter.export_cv_results(
                random_search, model_name, selector_name)

            # Log the results using ModelTraining's method
            self.log_search_results(random_search, model_name, selector_name)

            self.trained_model_info = {
                'pipeline': random_search.best_estimator_,
                'training_type': "RandomizedSearchCV",
                'hyperparameters': random_search.best_params_,
                'cv_score': random_search.best_score_
            }

        except Exception as e:
            self.log_parameter_error(self.logger, model_name, param_grid)
