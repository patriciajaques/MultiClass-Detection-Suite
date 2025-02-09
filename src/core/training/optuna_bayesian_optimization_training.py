import logging
from time import time
import traceback
import optuna
from optuna.samplers import TPESampler
from optuna.logging import set_verbosity, WARNING
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import pandas as pd

from core.reporting.metrics_reporter import MetricsReporter
from core.training.base_training import BaseTraining
from core.models.parameter_handlers.optuna_param_converter import OptunaParamConverter
from core.logging.logger_config import LoggerConfig, with_logging


class OptunaBayesianOptimizationTraining(BaseTraining):
    """
    Implementa otimização Bayesiana de hiperparâmetros usando Optuna.
    """

    def __init__(self):
        super().__init__()
        self.logger = LoggerConfig.get_logger('optuna_training')
        self.logger.setLevel(logging.DEBUG)

    def optimize_model(self, pipeline, model_name, model_params, selector_name,
                       X_train, y_train, n_iter, cv, scoring, n_jobs=-1,
                       selector_search_space=None, groups=None) -> None:
        """
        Otimiza hiperparâmetros usando Optuna e treina o modelo final.
        """
        
        self.logger.info(
            f"Otimizando {model_name} com Optuna e seletor {selector_name}")

        def objective(trial):
            """Função objetivo para o Optuna otimizar."""
            try:
                # Sugere hiperparâmetros do modelo e seletor
                model_hyperparams = OptunaParamConverter.suggest_parameters(
                    trial, model_params, model_name)
                selector_hyperparams = OptunaParamConverter.suggest_selector_hyperparameters(
                    trial, selector_search_space) if selector_search_space else {}

                # Configura pipeline com os hiperparâmetros sugeridos
                hyperparams = {**model_hyperparams, **selector_hyperparams}
                pipeline.set_params(**hyperparams)

                # Avalia performance usando validação cruzada
                cv_mean_score = cross_val_score(
                    estimator=pipeline,
                    X=X_train,
                    y=y_train,
                    scoring=scoring,
                    cv=cv,
                    groups=groups,
                    n_jobs=n_jobs
                ).mean()
                self.logger.info(
                    f"Trial completed successfully. Score: {cv_mean_score}")

                return cv_mean_score
            

            except Exception as e:
                self.logger.error(f"Cross-validation failed: {str(e)}")
                self.logger.error(f"Parameters that caused error: {hyperparams}")
                self.logger.error(traceback.format_exc())
                return float('-inf')

        # Configura e executa a otimização
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler()
        )

        study.optimize(objective, n_trials=n_iter)

        # Chama o método do ReportManager para exportar os trials
        MetricsReporter.export_trials(study, model_name, selector_name)

        # Treina modelo final com melhores hiperparâmetros
        best_pipeline = clone(pipeline)
        best_pipeline.set_params(**study.best_trial.params)
        best_pipeline.fit(X_train, y_train)

        # Registra resultados
        self.log_study_results(study, model_name, selector_name)

        # Armazena informações do modelo treinado
        self.trained_model_info = {
            'pipeline': best_pipeline,
            'training_type': "Optuna",
            'hyperparameters': study.best_trial.params,
            'cv_score': study.best_trial.value
        }

    def log_study_results(self, study: optuna.Study, model_name: str,
                          selector_name: str) -> None:
        """Registra resultados da otimização."""
        self.logger.info(f"Melhores parâmetros: {study.best_params}")
        self.logger.info(f"Melhor score CV: {study.best_value}")

        # Registra todas as combinações testadas
        self.logger.info(
            "Todas as combinações de hiperparâmetros e seus resultados:")
        trials_df = study.trials_dataframe()
        failed_trials = trials_df['value'].isna().sum()

        for trial in study.trials:
            if not pd.isna(trial.value):
                self.logger.info(
                    f"Parâmetros: {trial.params}, Score Médio: {trial.value}"
                )

        self.logger.info(
            f"Número de tentativas que resultaram em NaN para {model_name}: {failed_trials}"
        )
