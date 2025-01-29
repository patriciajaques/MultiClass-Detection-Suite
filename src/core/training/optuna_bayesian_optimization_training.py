from time import time
import optuna
from optuna.samplers import TPESampler
from optuna.logging import set_verbosity, WARNING
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import pandas as pd

from core.training.base_training import BaseTraining
from core.models.parameter_handlers.optuna_param_converter import OptunaParamConverter
from core.logging.logger_config import with_logging


@with_logging('optuna_training')
class OptunaBayesianOptimizationTraining(BaseTraining):
    """
    Implementa otimização Bayesiana de hiperparâmetros usando Optuna.
    """

    def __init__(self):
        super().__init__()

    def optimize_model(self, pipeline, model_name, model_params, selector_name,
                       X_train, y_train, n_iter, cv, scoring, n_jobs=-1,
                       selector_search_space=None) -> None:
        """
        Otimiza hiperparâmetros usando Optuna e treina o modelo final.
        """
        set_verbosity(WARNING)
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
                return cross_val_score(
                    estimator=pipeline,
                    X=X_train,
                    y=y_train,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=n_jobs
                ).mean()

            except Exception as e:
                self.log_parameter_error(model_name, hyperparams)
                return float('-inf')

        # Configura e executa a otimização
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler()
        )

        study.optimize(objective, n_trials=n_iter)

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
