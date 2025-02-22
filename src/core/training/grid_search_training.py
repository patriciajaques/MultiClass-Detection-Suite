"""
Copyright (c) 2025 Patricia Jaques
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License. To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from core.logging.logger_config import LoggerConfig, with_logging
from core.reporting.metrics_reporter import MetricsReporter
from core.training.base_training import BaseTraining
from core.models.parameter_handlers.grid_search_param_converter import GridSearchParamConverter


class GridSearchTraining(BaseTraining):
    def __init__(self):
        super().__init__()
        self.logger = LoggerConfig.get_logger('gridsearch_training')

    def optimize_model(self, pipeline, model_name, model_params, selector_name, X_train, y_train, n_iter, cv, scoring, n_jobs=-1, selector_search_space=None, groups=None):
        try:
            self.logger.info(
                f"Training and evaluating {model_name} with GridSearchCV and {selector_name}")

            param_grid = GridSearchParamConverter.convert_param_space(
                model_params, model_name)
            self.logger.info("Parameter grid before conversion:")
            self.logger.info(param_grid)

            if selector_search_space:
                param_grid = GridSearchParamConverter.combine_with_selector_space(
                    param_grid, selector_search_space)
            self.logger.info(
                "Final parameter grid after conversion chamando método GridSearchParamConverter.combine_with_selector_space:")
            self.logger.info(param_grid)
            

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                n_jobs=n_jobs,
                scoring=scoring,
                verbose=1,
                error_score=float('-inf')
            )

            grid_search.fit(X_train, y_train)

            # Chama a exportação dos resultados do CV (através do MetricsReporter)
            MetricsReporter.export_cv_results(
                grid_search, model_name, selector_name)

            # Log the results using ModelTraining's method
            self.log_search_results(
                grid_search, model_name, selector_name)

            self.trained_model_info = {
                'pipeline': grid_search.best_estimator_,
                'training_type': "GridSearchCV",
                'hyperparameters': grid_search.best_params_,
                'cv_score': grid_search.best_score_
            }

        except Exception as e:
            self.logger.warning(
                f"Parameter optimization failed for model {model_name}")
            self.logger.warning(f"Failed parameters configuration: {param_grid}")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.error("Full error traceback:", exc_info=True)
            self.trained_model_info = {
                'pipeline': None,
                'training_type': "GridSearchCV"
            }

            
