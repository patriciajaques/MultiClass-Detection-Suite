from sklearn.model_selection import GridSearchCV
from core.training.model_training import ModelTraining
from core.training.grid_search_model_params import GridSearchModelParams
import logging

class GridSearchTraining(ModelTraining):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

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

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            n_jobs=n_jobs,
            scoring=scoring,
            verbose=1
        )

        grid_search.fit(X_train, y_train)
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best cross-validation score: {grid_search.best_score_}")

        # Store the results
        self.trained_models[f"{model_name}_{selector_name}"] = {
            'model': grid_search.best_estimator_,
            'training_type': "GridSearchCV",
            'hyperparameters': grid_search.best_params_,
            'cv_result': grid_search.best_score_
        }