import optuna
from optuna.samplers import TPESampler
from optuna.logging import set_verbosity, WARNING
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from core.training.base_training import BaseTraining
from core.models.parameter_handlers.optuna_param_converter import OptunaParamConverter
from core.logging.logger_config import with_logging
from time import time
from sklearn.model_selection import train_test_split
import pandas as pd

@with_logging('optuna_training')
class OptunaBayesianOptimizationTraining(BaseTraining):
    def __init__(self):
        super().__init__()

    def optimize_model(self, pipeline, model_name, model_params, selector_name, X_train, y_train, n_iter, cv, scoring, n_jobs=-1, selector_search_space=None):
        set_verbosity(WARNING)
        self.logger.info(f"Training and evaluating {model_name} with Optuna Optimization and {selector_name}")
        print(f"Inside OptunaBayesianOptimizationTraining.optimize_model")


        def objective(trial):
            try:
                # Sugerir hiperparâmetros do modelo
                model_hyperparams = OptunaParamConverter.suggest_parameters(trial, model_params, model_name)
                 
                # Sugerir hiperparâmetros do seletor
                selector_hyperparams = OptunaParamConverter.suggest_selector_hyperparameters(
                    trial, selector_search_space) if selector_search_space else {}
                
                # Combinar os hiperparâmetros
                hyperparams = {**model_hyperparams, **selector_hyperparams}
                pipeline.set_params(**hyperparams)
                
                return cross_val_score(
                    estimator=pipeline,
                    X=X_train,
                    y=y_train,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=n_jobs
                ).mean()
                
            except Exception as e:
                self.log_parameter_error(self.logger, model_name, hyperparams)
                return float('-inf')

        # Criar um estudo do Optuna
        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        
        # Iniciar o tempo de otimização
        start_time = time()
        study.optimize(objective, n_trials=n_iter)
        total_time = time() - start_time

        # Criar uma cópia do pipeline para o treinamento final
        best_pipeline = clone(pipeline)

        # Configurar com os melhores hiperparâmetros
        best_pipeline.set_params(**study.best_trial.params)

        # Criar validação interna para early stopping
        X_train_main, X_valid, y_train_main, y_valid = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        # Treinar o pipeline final com todos os dados de treinamento
        best_pipeline.fit(
            X_train_main, y_train_main,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=10,
            eval_metric='mlogloss'
        )

        # Log the results using the overridden method
        self.log_search_results(self.logger, study, model_name, selector_name)
        
        # Armazenar os resultados
        self.trained_models[f"{model_name}_{selector_name}"] = {
            'model': best_pipeline,
            'training_type': "Optuna",
            'hyperparameters': study.best_trial.params,
            'cv_result': study.best_trial.value,
            'optimization_time_seconds': total_time
        }

        # Log final do melhor resultado
        self.logger.info(f"Optuna Optimization Best Result for {model_name} with {selector_name}: {study.best_trial.value}")
    
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
