from sklearn.model_selection import RandomizedSearchCV
from core.logging.logger_config import with_logging
from core.training.base_training import BaseTraining
from core.models.parameter_handlers.grid_search_param_converter import GridSearchParamConverter


@with_logging('random_search')
class RandomSearchTraining(BaseTraining):
    def __init__(self):
        super().__init__()

    def optimize_model(self, pipeline, model_name, model_params, selector_name, X_train, y_train, n_iter, cv, scoring, n_jobs=-1, selector_search_space=None):
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

            # Log the results using ModelTraining's method
            self.log_search_results(random_search, model_name, selector_name)

            self.trained_models[f"{model_name}_{selector_name}"] = {
                'pipeline': random_search.best_estimator_,
                'training_type': "RandomizedSearchCV",
                'hyperparameters': random_search.best_params_,
                'cv_score': random_search.best_score_
            }

        except Exception as e:
            self.log_parameter_error(self.logger, model_name, param_grid)
