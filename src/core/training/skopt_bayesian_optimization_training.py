from skopt import BayesSearchCV
from core.training.model_training import ModelTraining
from core.training.skopt_model_params import SkoptModelParams
from core.logging.logger_config import with_logging

@with_logging('skopt_training')
class SkoptBayesianOptimizationTraining(ModelTraining):
    def __init__(self):
        super().__init__()

    def optimize_model(self, pipeline, model_name, selector_name, X_train, y_train, n_iter, cv, scoring, n_jobs=-1, selector_search_space=None):
        try:
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

            search = BayesSearchCV(
                pipeline,
                search_spaces=search_space_model,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=42,
                verbose=0,
                error_score=float('-inf')  # Retorna -inf em vez de levantar erro
            )

            search.fit(X_train, y_train)
            
            # Log the results using ModelTraining's method
            self.log_search_results(self.logger, search, model_name, selector_name)

            self.trained_models[f"{model_name}_{selector_name}"] = {
                'model': search.best_estimator_,
                'training_type': "Bayesian Optimization",
                'hyperparameters': search.best_params_,
                'cv_result': search.best_score_
            }

            self.logger.info(f"Bayesian Optimization Best Result for {model_name} with {selector_name}: {search.best_score_}")
            print(f"Bayesian Optimization Best Result for {model_name} with {selector_name}: {search.best_score_}")

        except Exception as e:
            self.log_parameter_error(self.logger, model_name, search_space_model)

