from abc import ABC, abstractmethod
from time import time
from sklearn.pipeline import Pipeline

from typing import List, Optional, Dict, Any
import pandas as pd

from core.feature_selection.feature_selection_factory import FeatureSelectionFactory


class BaseTraining(ABC):
    def __init__(self):
        self.trained_models: Dict[str, Any] = {}
        self.total_execution_time = 0


    def train_model(
        self,
        X_train,
        y_train,
        model_params,  # define os parâmetros do modelo para o dataset específico
        selected_models: Optional[List[str]] = None, # Lista de nomes de modelos a serem treinados. Se None, usa todos os modelos.
        selected_selectors: Optional[List[str]] = None, # Lista de nomes de seletores a serem utilizados. Se None, usa todos os seletores.
        n_iter: int = 50, # Número de iterações para otimização.
        cv: int = 5, # Número de folds para validação cruzada.
        scoring: str = 'balanced_accuracy', # Métrica de avaliação.
        n_jobs: int = -1 # Número de trabalhos paralelos.
    ) -> Dict[str, Any]: # Dicionário contendo os modelos treinados e seus resultados.
        """
        Treina modelos com diferentes seletores de características.
        """

        start_time = time()

        available_selector_names = FeatureSelectionFactory.get_available_selectors_names()
        # Filtrar modelos
        models = self._filter_models(model_params.get_models(), selected_models)
        # Filtrar seletores
        selector_names = self._filter_selectors(selected_selectors, available_selector_names)

        for model_name, model_config in models.items():
            for selector_name in selector_names:
                if selector_name == 'none':
                    pipeline = self._create_pipeline(None, model_config)
                    selector_search_space = {}
                else:
                    # Criar a instância do seletor diretamente dentro do loop
                    selector_instance = FeatureSelectionFactory.create_selector(selector_name, X_train, y_train)
                    selector = selector_instance.selector  # Acessar o seletor criado no construtor

                    pipeline = self._create_pipeline(selector, model_config)
                    # Obter o espaço de busca diretamente do selector_instance
                    selector_search_space = selector_instance.get_search_space()

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

        self.total_execution_time = time() - start_time
        self._log_execution_time(len(models)) 
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
        steps = []
        if selector is not None:
            steps.append(('feature_selection', selector))
        steps.append(('classifier', model_config))
        return Pipeline(steps)

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
            selector_names = [s for s in selected_selectors if s in available_selector_names or s == 'none']
            missing_selectors = set(selected_selectors) - set(selector_names)
            if missing_selectors:
                raise ValueError(f"Seletores não encontrados: {missing_selectors}")
            return selector_names
        return available_selector_names + ['none']
    
    def log_parameter_error(self, logger, model_name: str, params: dict) -> None:
        """
        Método comum para logar erros de parâmetros inválidos.
        
        Args:
            logger: Logger configurado
            model_name: Nome do modelo
            params: Parâmetros que causaram o erro
        """
        logger.warning(f"Trial failed: Invalid parameter combination for {model_name}")
        logger.warning(f"Parameters that failed: {params}")
    
    @staticmethod
    def log_search_results(logger, search, model_name, selector_name):
        """Log the results of the search process."""
        if hasattr(search, 'best_params_'):
            logger.info(f"Best parameters: {search.best_params_}")
        if hasattr(search, 'best_score_'):
            logger.info(f"Best cross-validation score: {search.best_score_}")

        # Log all hyperparameter combinations and their cross-validation results
        logger.info("All hyperparameter combinations and their cross-validation results:")
        
        if hasattr(search, 'cv_results_'):
            cv_results = search.cv_results_
            success_count = 0
            for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
                if not pd.isna(mean_score):
                    success_count += 1
                    logger.info(f"Params: {params}, Mean Test Score: {mean_score}")
            
            logger.info(f"Number of successful trials for {model_name}: {success_count}")
            logger.info(f"Number of failed trials for {model_name}: {len(cv_results['mean_test_score']) - success_count}")

    def _log_execution_time(self, num_models):
        """
        Registra no log as informações sobre o tempo de execução do algoritmo.
        
        Args:
            num_models (int): Número total de modelos treinados
        """
        algorithm_name = self.__class__.__name__.replace('Training', '')
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Tempo total de execução do {algorithm_name}: {self.total_execution_time:.2f} segundos")
        self.logger.info(f"Média de tempo por modelo: {self.total_execution_time/num_models:.2f} segundos")
        self.logger.info(f"{'='*50}\n")