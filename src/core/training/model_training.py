from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline

from typing import List, Optional, Dict, Any
import logging
import pandas as pd

from core.logging.logger_config import LoggerConfig
from core.training.model_params import ModelParams
from core.feature_selection.feature_selection_factory import FeatureSelectionFactory


class ModelTraining(ABC):
    def __init__(self, logger_name):
        # Configurar um logger nomeado
        LoggerConfig.configure_log_file(
            file_main_name=logger_name,
            log_extension=".log",
            logger_name=logger_name
        )
        self.logger = logging.getLogger(logger_name)
        self.trained_models: Dict[str, Any] = {}

    def train_model(
        self,
        X_train,
        y_train,
        selected_models: Optional[List[str]] = None, # Lista de nomes de modelos a serem treinados.Se None, todos os modelos disponíveis serão utilizados.
        selected_selectors: Optional[List[str]] = None, # Lista de nomes de seletores a serem utilizados. Se None, todos os seletores disponíveis serão utilizados.
        n_iter: int = 50, # Número de iterações para otimização.
        cv: int = 5, # Número de folds para validação cruzada.
        scoring: str = 'balanced_accuracy', # Métrica de avaliação.
        n_jobs: int = -1 # Número de trabalhos paralelos.
    ) -> Dict[str, Any]: # Dicionário contendo os modelos treinados e seus resultados.
        """
        Treina modelos com diferentes seletores de características.
        """
        models = ModelParams.get_models()
        available_selector_names = FeatureSelectionFactory.get_available_selectors()

        # Filtrar modelos
        models = self._filter_models(models, selected_models)

        # Filtrar seletores
        selector_names = self._filter_selectors(selected_selectors, available_selector_names)

        for model_name, model_config in models.items():
            for selector_name in selector_names:
                # Criar a instância do seletor diretamente dentro do loop
                selector_instance = FeatureSelectionFactory.create_selector(selector_name, X_train, y_train)
                selector = selector_instance.selector  # Acessar o seletor criado no construtor

                pipeline = self._create_pipeline(selector, model_config)

                # Obter o espaço de busca diretamente do selector_instance
                selector_search_space = selector_instance.get_search_space()

                self.optimize_model(
                    pipeline,
                    model_name,
                    selector_name,
                    X_train,
                    y_train,
                    n_iter,
                    cv,
                    scoring,
                    n_jobs,
                    selector_search_space  # Passar o espaço de busca
                )

        return self.trained_models

    @abstractmethod
    def optimize_model(
        self,
        pipeline,
        model_name: str,
        selector_name: str,
        X_train,
        y_train,
        n_iter: int,
        cv: int,
        scoring: str,
        n_jobs: int,
        selector_search_space: dict
    ):
        pass

    @staticmethod
    def _create_pipeline(selector, model_config) -> Pipeline:
        """
        Cria um pipeline com seleção de características e o classificador.

        Args:
            selector: Seletor de características.
            model_config: Configuração do modelo de classificação.

        Returns:
            Pipeline: Pipeline configurado.
        """
        return Pipeline([
            ('feature_selection', selector),
            ('classifier', model_config)
        ])

    def _filter_models(self, models: Dict[str, Any], selected_models: Optional[List[str]]) -> Dict[str, Any]:
        """
        Filtra os modelos baseados na lista de modelos selecionados.

        Args:
            models (Dict[str, Any]): Dicionário de modelos disponíveis.
            selected_models (Optional[List[str]]): Lista de modelos a serem utilizados.

        Returns:
            Dict[str, Any]: Dicionário filtrado de modelos.

        Raises:
            ValueError: Se algum modelo selecionado não for encontrado.
        """
        if selected_models is not None:
            filtered_models = {name: cfg for name, cfg in models.items() if name in selected_models}
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
            selector_names = [s for s in selected_selectors if s in available_selector_names]
            missing_selectors = set(selected_selectors) - set(selector_names)
            if missing_selectors:
                raise ValueError(f"Seletores não encontrados: {missing_selectors}")
            return selector_names
        return available_selector_names
    
    @staticmethod
    def log_search_results(logger, search, model_name, selector_name):
        """Log the results of the search process (GridSearchCV, RandomizedSearchCV, or any search with cv_results_)."""
        if hasattr(search, 'best_params_'):
            logger.info(f"Best parameters: {search.best_params_}")
        if hasattr(search, 'best_score_'):
            logger.info(f"Best cross-validation score: {search.best_score_}")

        # Log all hyperparameter combinations and their cross-validation results
        logger.info("All hyperparameter combinations and their cross-validation results:")
        if hasattr(search, 'cv_results_'):
            cv_results = search.cv_results_
            nan_count = 0
            for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
                if pd.isna(mean_score):
                    nan_count += 1
                logger.info(f"Params: {params}, Mean Test Score: {mean_score}")
            logger.info(f"Number of tests that resulted in NaN for {model_name}: {nan_count}")
