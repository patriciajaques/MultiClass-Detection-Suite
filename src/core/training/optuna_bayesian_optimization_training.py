import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from core.training.model_training import ModelTraining
from core.training.optuna_model_params import OptunaModelParams
from core.logging.logger_config import LoggerConfig
import logging
from time import time
import pandas as pd

class OptunaBayesianOptimizationTraining(ModelTraining):
    def __init__(self):
        super().__init__(logger_name='optuna_training')

    def optimize_model(self, pipeline, model_name, selector_name, X_train, y_train, n_trials, cv, scoring, n_jobs=-1, selector_search_space=None):
        self.logger.info(f"Training and evaluating {model_name} with Optuna Bayesian Optimization and {selector_name}")
        print(f"Training and evaluating {model_name} with Optuna Bayesian Optimization and {selector_name}")

        def objective(trial):
            try:
                model_hyperparams = OptunaModelParams.suggest_hyperparameters(trial, model_name)
                
                # Sugerir hiperparâmetros para o seletor de features
                selector_hyperparams = {}
                for param, values in selector_search_space.items():
                    if isinstance(values, list) and isinstance(values[0], int):
                        selector_hyperparams[param] = trial.suggest_int(param, min(values), max(values))
                    elif isinstance(values, list) and isinstance(values[0], float):
                        selector_hyperparams[param] = trial.suggest_float(param, min(values), max(values))
                    elif isinstance(values, list) and isinstance(values[0], str):
                        selector_hyperparams[param] = trial.suggest_categorical(param, values)
                    else:
                        selector_hyperparams[param] = trial.suggest_categorical(param, values)
                
                # Combinar os hiperparâmetros do modelo e do seletor
                hyperparams = {**model_hyperparams, **selector_hyperparams}
                
                pipeline.set_params(**hyperparams)
                
                score = cross_val_score(
                    estimator=pipeline,
                    X=X_train,
                    y=y_train,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=n_jobs
                ).mean()
                
                return score
            except Exception as e:
                self.logger.warning(f"Trial failed with error: {str(e)}")
                return float('-inf')  # Retorna um valor muito baixo para indicar que o trial falhou
        
        # Criar um estudo do Optuna
        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        
        # Iniciar o tempo de otimização
        start_time = time()
        study.optimize(objective, n_trials=n_trials)
        total_time = time() - start_time

        # Atualizar o pipeline com os melhores hiperparâmetros
        pipeline.set_params(**study.best_trial.params)
        
        # Treinar o modelo final com os melhores hiperparâmetros
        pipeline.fit(X_train, y_train)

        # Log the results using the overridden method
        self.log_search_results(self.logger, study, model_name, selector_name)
        
        # Armazenar os resultados
        self.trained_models[f"{model_name}_{selector_name}"] = {
            'model': pipeline,
            'training_type': "Optuna Bayesian Optimization",
            'hyperparameters': study.best_trial.params,
            'cv_result': study.best_trial.value,
            'optimization_time_seconds': total_time
        }

        # Log final do melhor resultado
        self.logger.info(f"Optuna Optimization Best Result for {model_name} with {selector_name}: {study.best_trial.value}")
        print(f"Optuna Optimization Best Result for {model_name} with {selector_name}: {study.best_trial.value}")
    
    @staticmethod
    def log_search_results(logger, study, model_name, selector_name):
        """Log the results of the Optuna optimization process."""
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best cross-validation score: {study.best_value}")

        # Log all hyperparameter combinations and their cross-validation results
        logger.info("All hyperparameter combinations and their cross-validation results:")
        nan_count = 0
        for trial in study.trials:
            mean_score = trial.value
            params = trial.params
            if pd.isna(mean_score):
                nan_count += 1
            logger.info(f"Params: {params}, Mean Test Score: {mean_score}")
        logger.info(f"Number of tests that resulted in NaN for {model_name}: {nan_count}")
