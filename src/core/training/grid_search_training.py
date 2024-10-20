import pandas as pd
from sklearn.model_selection import GridSearchCV
from core.logging.logger_config import LoggerConfig
from core.training.model_training import ModelTraining
from core.training.grid_search_model_params import GridSearchModelParams
import logging


class GridSearchTraining(ModelTraining):
    def __init__(self):
        super().__init__(logger_name='grid_search_training')

    def optimize_model(self, pipeline, model_name, selector_name, X_train, y_train, n_iter, cv, scoring, n_jobs=-1, selector_search_space=None):
        self.logger.info(f"Training and evaluating {model_name} with GridSearchCV and {selector_name}")

        # Combine model and selector parameter grids
        param_grid = GridSearchModelParams.get_param_grid(model_name)
        if selector_search_space:
            if isinstance(param_grid, list):
                for subspace in param_grid:
                    subspace.update(selector_search_space)
            else:
                param_grid.update(selector_search_space)

        self.logger.debug(f"Parameter grid: {param_grid}")

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            n_jobs=n_jobs,
            scoring=scoring,
            verbose=1
        )

        self.logger.info("Starting GridSearchCV fitting process")
        grid_search.fit(X_train, y_train)
        self.logger.info("GridSearchCV fitting process completed")

        # Passar os argumentos necessários para log_search_results
        self.log_search_results(self.logger, grid_search, model_name, selector_name)

        # Store the results
        self.trained_models[f"{model_name}_{selector_name}"] = {
            'model': grid_search.best_estimator_,
            'training_type': "GridSearchCV",
            'hyperparameters': grid_search.best_params_,
            'cv_result': grid_search.best_score_
        }
        self.logger.info(f"Model {model_name}_{selector_name} stored successfully")

    # def _log_grid_search_results(self, grid_search, model_name, selector_name):
    #     """Log the results of the GridSearchCV process."""
    #     self.logger.info(f"Best parameters: {grid_search.best_params_}")
    #     self.logger.info(f"Best cross-validation score: {grid_search.best_score_}")

    #     # Log all hyperparameter combinations and their cross-validation results
    #     self.logger.info("All hyperparameter combinations and their cross-validation results:")
    #     cv_results = grid_search.cv_results_
    #     nan_count = 0
    #     for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
    #         if pd.isna(mean_score):
    #             nan_count += 1
    #         self.logger.info(f"Params: {params}, Mean Test Score: {mean_score}")

    #     self.logger.info(f"Number of tests that resulted in NaN for {model_name}: {nan_count}")