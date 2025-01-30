from abc import ABC, abstractmethod
import pandas as pd
from sklearn.pipeline import Pipeline

from core.config.config_manager import ConfigManager
from core.preprocessors.data_balancer import DataBalancer
from core.preprocessors.data_cleaner import DataCleaner
from core.reporting.feature_mapping_reporter import FeatureMappingReporter
from core.reporting.report_formatter import ReportFormatter
from core.utils.path_manager import PathManager

class BasePipeline(ABC):
    """Classe base abstrata para pipelines de detecção."""

    def __init__(self, target_column: str, n_iter=50, n_jobs=6, test_size=0.2):
        """
        Inicializa o pipeline base.
        
        Args:
            n_iter (int): Número de iterações para otimização
            n_jobs (int): Número de jobs paralelos
            test_size (float): Tamanho do conjunto de teste
            config_dir (str): Diretório de configurações
        """
        self.target_column = target_column
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.test_size = test_size
        self.paths = {
            'data': PathManager.get_path('data'),
            'output': PathManager.get_path('output'),
            'models': PathManager.get_path('models'),
            'src': PathManager.get_path('src')
        }
        self.config = ConfigManager() 
        self.model_params = self._get_model_params()
        self.data_cleaner = DataCleaner(config_manager=self.config)
        self.X_encoder = None
        self.y_encoder = None
        ReportFormatter.setup_formatting(4)

    @staticmethod
    def create_pipeline(selector, model_config) -> Pipeline:
        """
        Cria um pipeline com seleção de características e classificador.
        """
        steps = []
        if selector is not None:
            # Remover qualquer parâmetro 'selector' que possa existir
            if hasattr(selector, 'selector'):
                delattr(selector, 'selector')
            steps.append(('feature_selection', selector))
        steps.append(('classifier', model_config))
        return Pipeline(steps)

    def _get_training_stages(self):
        """Define os stages (algoritmo e seletor) de treinamento usando configuração."""
        try:
            training_config = self.config.get_config('training_settings')

            models = training_config.get('models', ['Naive Bayes'])
            selectors = training_config.get('selectors', ['none'])

            stages = []

            for model in models:
                for selector in selectors:
                    stages.append((model, selector))

            return stages

        except Exception as e:
            print(f"Erro ao carregar configurações de treinamento: {str(e)}")
            return [('naive_bayes_none', ['Naive Bayes'], ['none'])]

    @abstractmethod
    def _get_model_params(self):
        """Retorna os parâmetros do modelo específico do pipeline."""
        pass

    @abstractmethod
    def load_and_clean_data(self):
        """Carrega e prepara os dados específicos do pipeline."""
        pass

    @abstractmethod
    def prepare_data(self, data):
        """Prepara os dados para treinamento."""
        pass

    def _verify_split_quality(self, train_data, test_data, tolerance: float = 0.15):
        """
        Verifica se o split manteve as proporções de classes desejadas dentro da tolerância especificada.

        Args:
            train_data (pd.DataFrame): Dados de treino
            test_data (pd.DataFrame): Dados de teste
            tolerance (float): Tolerância máxima permitida para diferença na distribuição (0.0 a 1.0)

        Raises:
            ValueError: Se os dados não contiverem a coluna target
        """
        if self.target_column not in train_data.columns or self.target_column not in test_data.columns:
            raise ValueError(
                f"Coluna target '{self.target_column}' não encontrada nos dados")

        # Calcula distribuições
        train_dist = train_data[self.target_column].value_counts(normalize=True)
        test_dist = test_data[self.target_column].value_counts(normalize=True)

        # Verifica distribuição para cada classe
        for class_name in train_dist.index:
            train_prop = train_dist[class_name]
            # Usa 0 se a classe não existir no teste
            test_prop = test_dist.get(class_name, 0)
            diff = abs(train_prop - test_prop)

            if diff >= tolerance:
                print(
                    f"Aviso: Diferença significativa detectada para '{class_name}' "
                    f"(treino: {train_prop:.2%}, teste: {test_prop:.2%}, "
                    f"diferença: {diff:.2%}, tolerância: {tolerance:.2%})"
                )

    def balance_data(self, X_train, y_train, strategy='auto'):
        print("\nIniciando balanceamento de dados...")
        print(f"Tipo de X_train: {type(X_train)}")
        print(f"Tipo de y_train: {type(y_train)}")
        print(f"Shape de X_train antes do balanceamento: {X_train.shape}")
        print(f"Distribuição de classes antes do balanceamento:")
        print(y_train.value_counts())

        # Garantir que os dados estão no formato correto
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        data_balancer = DataBalancer()

        # Na função balance_data do BasePipeline
        strategy = 0.75  # Gera 75% do número de amostras da classe majoritária
        X_resampled, y_resampled = data_balancer.apply_smote(
            X_train, y_train, strategy=strategy)

        print(f"Shape de X_train após balanceamento: {X_resampled.shape}")
        print(f"Distribuição de classes após balanceamento:")
        print(pd.Series(y_resampled).value_counts())

        return X_resampled, y_resampled

    def run(self):
        from core.management.stage_training_manager import StageTrainingManager

        """Executa o pipeline completo."""
        print(f"Iniciando pipeline de {self.__class__.__name__}...")

        # Load and prepare data
        print("\n1. Carregando e preparando dados...")
        data = self.load_and_clean_data()

        # Prepare data
        print("\n2. Preparando dados para treinamento...")
        X_train, X_test, y_train, y_test = self.prepare_data(data)

        # Gerando report das features
        feature_report = FeatureMappingReporter()
        feature_report.log_feature_mappings(self.X_encoder)
        feature_report.log_target_mappings(self.y_encoder)
        
        # Balance data
        print("\n3. Balanceando dados de treino...")
        X_train, y_train = self.balance_data(X_train, y_train, strategy='auto')

        # Train models
        print("\n4. Iniciando treinamento dos modelos...")
        training_manager = StageTrainingManager(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_params=self.model_params,
            n_iter=self.n_iter,
            cv=10,
            scoring='balanced_accuracy',
            n_jobs=self.n_jobs)

        # Define training stages
        stages = self._get_training_stages()

        # Execute training stages
        training_manager.execute_all_stages(stages)

        print("\nPipeline concluído!")
