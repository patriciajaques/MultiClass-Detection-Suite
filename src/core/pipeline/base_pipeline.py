from abc import ABC, abstractmethod
import logging
from typing import Optional
import pandas as pd
from sklearn.pipeline import Pipeline

from core.config.config_manager import ConfigManager
from core.logging.logger_config import LoggerConfig
from core.preprocessors.data_balancer import DataBalancer
from core.preprocessors.data_cleaner import DataCleaner
from core.reporting.feature_mapping_reporter import FeatureMappingReporter
from core.reporting.report_formatter import ReportFormatter
from core.utils.path_manager import PathManager


from core.training.random_search_training import RandomSearchTraining
from core.training.optuna_bayesian_optimization_training import OptunaBayesianOptimizationTraining
from core.training.grid_search_training import GridSearchTraining


class BasePipeline(ABC):
    """Classe base abstrata para pipelines de detecção."""

    def __init__(self, target_column: str, n_iter=50, n_jobs=6, val_size=None, test_size=0.2, training_strategy_name='optuna', use_voting_classifier=True):
        """
        Args:
            target_column: Nome da coluna alvo
            n_iter: Número de iterações para otimização
            n_jobs: Número de jobs paralelos
            val_size: Tamanho da validação
            test_size: Tamanho do teste
            training_strategy: Estratégia de otimização ('optuna', 'random', 'grid')
        """
        # Configurar logger no início
        self.logger = LoggerConfig.get_logger('preprocessing')
        self.logger.info("Iniciando pipeline...")

        self.target_column = target_column
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.val_size = val_size
        self.test_size = test_size

        self.paths = {
            'data': PathManager.get_path('data'),
            'output': PathManager.get_path('output'),
            'models': PathManager.get_path('models'),
            'src': PathManager.get_path('src')
        }
        self.config_manager = ConfigManager()
        self.model_params = self._get_model_params()
        self.data_cleaner = DataCleaner()
        self.X_encoder = None
        self.y_encoder = None
        ReportFormatter.setup_formatting(4)
        self.training_strategy = self._initialize_training_manager(
            training_strategy_name)
        self.use_voting_classifier = use_voting_classifier

    def _initialize_training_manager(self, strategy: str):
        """Inicializa o gerenciador de treinamento apropriado."""
        strategies = {
            'optuna': OptunaBayesianOptimizationTraining,
            'random': RandomSearchTraining,
            'grid': GridSearchTraining
        }

        if strategy not in strategies:
            self.logger.warning(
                f"Estratégia {strategy} não encontrada. Usando optuna como fallback.")
            return OptunaBayesianOptimizationTraining()

        return strategies[strategy]()


    @staticmethod
    def create_pipeline(selector, model_config_manager) -> Pipeline:
        """
        Cria um pipeline com seleção de características e classificador.
        """
        steps = []
        if selector is not None:
            # Remover qualquer parâmetro 'selector' que possa existir
            if hasattr(selector, 'selector'):
                delattr(selector, 'selector')
            steps.append(('feature_selection', selector))
        steps.append(('classifier', model_config_manager))
        return Pipeline(steps)

    def _get_training_stages(self):
        """Define os stages (algoritmo e seletor) de treinamento usando config_manageruração."""
        try:
            training_config_manager = self.config_manager.get_config('training_settings')

            models = training_config_manager.get('models', ['Naive Bayes'])
            selectors = training_config_manager.get('selectors', ['none'])

            stages = []

            for model in models:
                for selector in selectors:
                    stages.append((model, selector))

            return stages

        except Exception as e:
            self.logger.info(
                f"Erro ao carregar configurações de treinamento: {str(e)}")
            return [['Naive Bayes'], ['none']]

    @abstractmethod
    def _get_model_params(self):
        """Retorna os parâmetros do modelo específico do pipeline."""
        pass

    @abstractmethod
    def load_data(self):
        """Carrega e prepara os dados específicos do pipeline."""
        pass

    @abstractmethod
    def clean_data(self, data):
        """Carrega e prepara os dados específicos do pipeline."""
        pass

    @abstractmethod
    def prepare_data(self, data):
        """Prepara os dados para treinamento."""
        pass

    def _verify_split_quality(self, train_data, val_data, test_data, tolerance: float = 0.15):
        """
        Verifica se o split manteve as proporções de classes desejadas dentro da tolerância especificada.

        Args:
            train_data (pd.DataFrame): Dados de treino
            val_data (pd.DataFrame): Dados de validação
            test_data (pd.DataFrame): Dados de teste
            tolerance (float): Tolerância máxima permitida para diferença na distribuição (0.0 a 1.0)

        Raises:
            ValueError: Se os dados não contiverem a coluna target
        """
        if self.target_column not in train_data.columns or self.target_column not in val_data.columns or self.target_column not in test_data.columns:
            raise ValueError(
                f"Coluna target '{self.target_column}' não encontrada nos dados")

        # Calcula distribuições
        train_dist = train_data[self.target_column].value_counts(
            normalize=True)
        val_dist = val_data[self.target_column].value_counts(normalize=True)
        test_dist = test_data[self.target_column].value_counts(normalize=True)

        # Verifica distribuição para cada classe
        for class_name in train_dist.index:
            train_prop = train_dist[class_name]
            val_prop = val_dist.get(class_name, 0)
            test_prop = test_dist.get(class_name, 0)

            val_diff = abs(train_prop - val_prop)
            test_diff = abs(train_prop - test_prop)

            if val_diff >= tolerance:
                self.logger.info(
                    f"Aviso: Diferença significativa detectada para '{class_name}' "
                    f"(treino: {train_prop:.2%}, validação: {val_prop:.2%}, "
                    f"diferença: {val_diff:.2%}, tolerância: {tolerance:.2%})"
                )

            if test_diff >= tolerance:
                self.logger.info(
                    f"Aviso: Diferença significativa detectada para '{class_name}' "
                    f"(treino: {train_prop:.2%}, teste: {test_prop:.2%}, "
                    f"diferença: {test_diff:.2%}, tolerância: {tolerance:.2%})"
                )


    def _verify_stratified_split(self,data: pd.DataFrame,
                                train_data: pd.DataFrame,
                                val_data: Optional[pd.DataFrame],
                                test_data: pd.DataFrame,
                                target_column: str,
                                tolerance: float = 0.05) -> None:
        """
        Verifica se o split estratificado manteve as proporções das classes.

        Args:
            data: DataFrame original
            train_data: DataFrame de treino
            val_data: DataFrame de validação (opcional)
            test_data: DataFrame de teste
            target_column: Nome da coluna target
            tolerance: Tolerância máxima para diferença nas proporções
        """
        self.logger.info("\nVerificando distribuição das classes:")

        # Calcula proporções
        orig_dist = data[target_column].value_counts(normalize=True)
        train_dist = train_data[target_column].value_counts(normalize=True)
        test_dist = test_data[target_column].value_counts(normalize=True)

        self.logger.info("\nDistribuição original:")
        self.logger.info(orig_dist)
        self.logger.info("\nDistribuição treino:")
        self.logger.info(train_dist)

        if val_data is not None:
            val_dist = val_data[target_column].value_counts(normalize=True)
            self.logger.info("\nDistribuição validação:")
            self.logger.info(val_dist)

        self.logger.info("\nDistribuição teste:")
        self.logger.info(test_dist)

        # Verifica diferenças
        self.logger.info("\nVerificando diferenças nas proporções...")
        max_diff_train = max(abs(orig_dist - train_dist))
        max_diff_test = max(abs(orig_dist - test_dist))

        self.logger.info(f"Máxima diferença no treino: {max_diff_train:.4f}")
        self.logger.info(f"Máxima diferença no teste: {max_diff_test:.4f}")

        if val_data is not None:
            max_diff_val = max(abs(orig_dist - val_dist))
            self.logger.info(
                f"Máxima diferença na validação: {max_diff_val:.4f}")

        # Alerta se diferenças excedem tolerância
        if max_diff_train > tolerance or max_diff_test > tolerance:
            self.logger.info(
                "\nALERTA: Diferenças nas proporções excedem a tolerância!")

        # Verifica tamanhos dos conjuntos
        self.logger.info("\nTamanho dos conjuntos:")
        self.logger.info(f"Original: {len(data)}")
        self.logger.info(
            f"Treino: {len(train_data)} ({len(train_data)/len(data):.2%})")
        if val_data is not None:
            self.logger.info(
                f"Validação: {len(val_data)} ({len(val_data)/len(data):.2%})")
        self.logger.info(
            f"Teste: {len(test_data)} ({len(test_data)/len(data):.2%})")

    def run(self):
        from core.management.stage_training_manager import StageTrainingManager

        """Executa o pipeline completo."""
        self.logger.info(f"Iniciando pipeline de {self.__class__.__name__}...")

        # Load and prepare data
        self.logger.info("\n1. Carregando  dados...")
        data = self.load_data()

        # Load and prepare data
        self.logger.info("\n1. Limpando dados...")
        data = self.clean_data(data)

        # Prepare data
        self.logger.info("\n2. Preparando dados para treinamento...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(
            data)
        self.logger.info(f"X_train shape: {X_train.shape}")
        self.logger.info(f"X_val shape: {X_val.shape}")
        self.logger.info(f"X_test shape: {X_test.shape}")

        # Gerando report das features
        feature_report = FeatureMappingReporter()
        feature_report.log_feature_mappings(self.X_encoder, X=data.drop(columns=[self.target_column]))
        feature_report.log_numeric_feature_mappings(X=data.drop(columns=[self.target_column]))
        feature_report.log_target_mappings(self.y_encoder, y=data[self.target_column])

        # Balance data
        self.logger.info("\n3. Balanceando dados de treino...")
        self.logger.info(f"X_train shape antes balanceamento: {X_train.shape}")
        self.logger.info(
            f"Distribuição original das classes:\n{y_train.value_counts()}")
        X_train, y_train = DataBalancer().balance_data(X_train, y_train, strategy='auto')
        self.logger.info(f"X_train shape após balanceamento: {X_train.shape}")
        self.logger.info(
            f"Distribuição após balanceamento das classes:\n{y_train.value_counts()}")

        # Train models
        self.logger.info("\n4. Iniciando treinamento dos modelos...")
        training_manager = StageTrainingManager(
            X_train=X_train,
            X_test=X_test,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            model_params=self.model_params,
            n_iter=self.n_iter,
            cv=10,
            group_feature='aluno',
            scoring='balanced_accuracy',
            n_jobs=self.n_jobs,
            training_strategy=self.training_strategy,
            use_voting_classifier=self.use_voting_classifier)

        # Define training stages
        stages = self._get_training_stages()

        # Execute training stages
        training_manager.execute_all_stages(stages)

        self.logger.info("\nPipeline concluído!")
