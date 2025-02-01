import pandas as pd
from sklearn.model_selection import GridSearchCV
from core.logging.logger_config import LoggerConfig, with_logging
from core.training.base_training import BaseTraining
from core.models.parameter_handlers.grid_search_param_converter import GridSearchParamConverter


class GridSearchTraining(BaseTraining):
    def __init__(self):
        super().__init__()
        self.logger = LoggerConfig.get_logger('gridsearch_training')

    def optimize_model(self, pipeline, model_name, model_params, selector_name, X_train, y_train, n_iter, cv, scoring, n_jobs=-1, selector_search_space=None):
        try:
            self.logger.info(
                f"Training and evaluating {model_name} with GridSearchCV and {selector_name}")

            param_grid = GridSearchParamConverter.convert_param_space(
                model_params, model_name)
            if selector_search_space:
                if isinstance(param_grid, list):
                    for subspace in param_grid:
                        subspace.update(selector_search_space)
                else:
                    param_grid.update(selector_search_space)

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                n_jobs=n_jobs,
                scoring=scoring,
                verbose=0,
                # Retorna -inf em vez de levantar erro
                error_score=float('-inf')
            )

            grid_search.fit(X_train, y_train)

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
            self.log_parameter_error(model_name, param_grid)
