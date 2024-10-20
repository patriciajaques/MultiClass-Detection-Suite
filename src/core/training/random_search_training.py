from sklearn.model_selection import RandomizedSearchCV
from core.training.model_training import ModelTraining
from core.training.grid_search_model_params import GridSearchModelParams

class RandomSearchTraining(ModelTraining):
    def __init__(self):
        super().__init__(logger_name='random_search_training')

    def optimize_model(self, pipeline, model_name, selector_name, X_train, y_train, n_iter, cv, scoring, n_jobs=-1, selector_search_space=None):
        self.logger.info(f"Training and evaluating {model_name} with RandomizedSearchCV and {selector_name}")

        # Combine model and selector parameter grids
        param_grid = GridSearchModelParams.get_param_grid(model_name)
        if selector_search_space:
            if isinstance(param_grid, list):
                for subspace in param_grid:
                    subspace.update(selector_search_space)
            else:
                param_grid.update(selector_search_space)

        self.logger.debug(f"Parameter grid: {param_grid}")

        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            n_jobs=n_jobs,
            scoring=scoring,
            verbose=1,
            random_state=42  # Para reprodutibilidade
        )

        self.logger.info("Starting RandomizedSearchCV fitting process")
        random_search.fit(X_train, y_train)
        self.logger.info("RandomizedSearchCV fitting process completed")

        # Log the results using ModelTraining's method
        self.log_search_results(self.logger, random_search, model_name, selector_name)

        self.logger.info(f"Best parameters: {random_search.best_params_}")
        self.logger.info(f"Best cross-validation score: {random_search.best_score_}")

        # Store the results
        self.trained_models[f"{model_name}_{selector_name}"] = {
            'model': random_search.best_estimator_,
            'training_type': "RandomizedSearchCV",
            'hyperparameters': random_search.best_params_,
            'cv_result': random_search.best_score_
        }
        self.logger.info(f"Model {model_name}_{selector_name} stored successfully")