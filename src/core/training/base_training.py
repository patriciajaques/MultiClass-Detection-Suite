from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd

from core.feature_selection.feature_selection_factory import FeatureSelectionFactory


from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
from sklearn.pipeline import Pipeline
from time import time


class BaseTraining(ABC):
    """
    Classe base abstrata para diferentes estratégias de treinamento.
    Implementa o padrão Template Method para treinar um único pipeline.
    """

    def __init__(self):
        self.trained_model_info = None
        self.execution_time = 0

    def train_model(self,
                    pipeline: Pipeline,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    model_name: str,
                    model_params: Any,
                    selector_name: str,
                    n_iter: int = 50,
                    cv: int = 5,
                    scoring: str = 'balanced_accuracy',
                    n_jobs: int = -1,
                    selector_search_space: Dict = None) -> Dict[str, Any]:
        """
        Treina um único pipeline com uma combinação específica de modelo e seletor.
        Mantém registro do tempo de execução.
        """
        try:
            start_time = time()

            # Otimiza e treina o modelo
            self.optimize_model(
                pipeline=pipeline,
                model_name=model_name,
                model_params=model_params,
                selector_name=selector_name,
                X_train=X_train,
                y_train=y_train,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                selector_search_space=selector_search_space
            )

            self.execution_time = time() - start_time
            self._log_execution_time(model_name, selector_name)

            # Adiciona tempo de execução às informações do modelo
            if self.trained_model_info:
                self.trained_model_info['execution_time'] = self.execution_time

            return self.trained_model_info

        except Exception as e:
            self.log_training_error(e, model_name, selector_name)
            return None

    @abstractmethod
    def optimize_model(self,
                       pipeline: Pipeline,
                       model_name: str,
                       model_params: Any,
                       selector_name: str,
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       n_iter: int,
                       cv: int,
                       scoring: str,
                       n_jobs: int,
                       selector_search_space: Dict) -> None:
        """
        Método abstrato para otimização de hiperparâmetros.
        Deve ser implementado por cada estratégia específica de treinamento.
        """
        pass

    def _log_execution_time(self, model_name: str, selector_name: str) -> None:
        """
        Registra o tempo de execução do treinamento.
        
        Args:
            model_name: Nome do modelo treinado
            selector_name: Nome do seletor de features usado
        """
        self.logger.info(f"\n{'='*50}")
        self.logger.info(
            f"Tempo total de execução para {model_name} com {selector_name}: "
            f"{self.execution_time:.2f} segundos"
        )
        self.logger.info(f"{'='*50}\n")

    def log_training_error(self, error: Exception, model_name: str, selector_name: str) -> None:
        """Registra erros ocorridos durante o treinamento."""
        self.logger.error(
            f"Erro ao treinar {model_name} com {selector_name}: {str(error)}")
        import traceback
        self.logger.error(traceback.format_exc())


    def _filter_models(self, models: Dict[str, Any], selected_models: Optional[List[str]]) -> Dict[str, Any]:
        """
        Filtra os modelos baseados na lista de modelos selecionados.

        Args:
            models (Dict[str, Any]): Dicionário de modelos disponíveis.
            selected_models (Optional[List[str]]): Lista de modelos a serem utilizados.

        Returns:
            Dict[str, Any]: Dicionário filtrado de modelos.
        """
        if selected_models is not None:
            filtered_models = {name: cfg for name,
                               cfg in models.items() if name in selected_models}
            missing_models = set(selected_models) - set(filtered_models.keys())
            if missing_models:
                raise ValueError(f"Modelos não encontrados: {missing_models}")
            return filtered_models
        return models

    def _filter_selectors(self, selected_selectors: Optional[List[str]], available_selector_names: List[str]) -> List[str]:
        """
        Filtra os seletores baseados na lista de seletores selecionados.

        Args:
            selected_selectors (Optional[List[str]]): Lista de seletores a serem utilizados.
            available_selector_names (List[str]): Lista de seletores disponíveis.

        Returns:
            List[str]: Lista filtrada de seletores.

        Raises:
            ValueError: Se algum seletor selecionado não for encontrado.
        """
        if selected_selectors is not None:
            selector_names = [
                s for s in selected_selectors if s in available_selector_names or s == 'none']
            missing_selectors = set(selected_selectors) - set(selector_names)
            if missing_selectors:
                raise ValueError(
                    f"Seletores não encontrados: {missing_selectors}")
            return selector_names
        return available_selector_names + ['none']

    def log_parameter_error(self, model_name: str, params: dict) -> None:
        """
        Logs parameter-related errors during model optimization.
        
        Args:
            model_name: Name of the model that encountered the error
            params: Parameters that caused the error
        """
        self.logger.warning(
            f"Parameter optimization failed for model {model_name}")
        self.logger.warning(f"Failed parameters configuration: {params}")

    @staticmethod
    def log_search_results(logger, search, model_name, selector_name):
        """Log the results of the search process."""
        if hasattr(search, 'best_params_'):
            logger.info(f"Best parameters: {search.best_params_}")
        if hasattr(search, 'best_score_'):
            logger.info(f"Best cross-validation score: {search.best_score_}")

        # Log all hyperparameter combinations and their cross-validation results
        logger.info(
            "All hyperparameter combinations and their cross-validation results:")

        if hasattr(search, 'cv_results_'):
            cv_results = search.cv_results_
            success_count = 0
            for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
                if not pd.isna(mean_score):
                    success_count += 1
                    logger.info(
                        f"Params: {params}, Mean Test Score: {mean_score}")

            logger.info(
                f"Number of successful trials for {model_name}: {success_count}")
            logger.info(
                f"Number of failed trials for {model_name}: {len(cv_results['mean_test_score']) - success_count}")