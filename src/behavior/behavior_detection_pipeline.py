import numpy as np
import pandas as pd

from behavior.behavior_data_loader import BehaviorDataLoader
from behavior.temporal_features_processor import TemporalFeaturesProcessor
from core.logging.feature_mapping_logger import FeatureMappingLogger
from core.preprocessors.data_cleaner import DataCleaner
from core.preprocessors.data_encoder import DataEncoder
from core.preprocessors.data_imputer import DataImputer
from core.preprocessors.data_splitter import DataSplitter
from behavior.behavior_model_params import BehaviorModelParams
from core.management.stage_training_manager import StageTrainingManager

from core.pipeline.base_pipeline import BasePipeline
from behavior.behavior_model_params import BehaviorModelParams


class BehaviorDetectionPipeline(BasePipeline):
    def __init__(self, n_iter=50, n_jobs=6, test_size=0.2):
        """
        Inicializa o pipeline de detecção de comportamentos.

        Args:
            target_column (str): Nome da coluna alvo
            n_iter (int): Número de iterações para otimização de hiperparâmetros
            n_jobs (int): Número de jobs paralelos para processamento
            test_size (float): Proporção dos dados para conjunto de teste
        """
        super().__init__(
            target_column='comportamento',
            n_iter=n_iter,
            n_jobs=n_jobs,
            test_size=test_size,
        )
        self.data_cleaner = DataCleaner(config_manager=self.config)
        
    def _get_model_params(self):
        """Obtém os parâmetros do modelo de comportamento."""
        return BehaviorModelParams()

    def load_and_clean_data(self):
        """Carrega e limpa o dataset."""
        # Load data
        data = BehaviorDataLoader.load_data(
            self.paths['data'] / 'new_logs_labels.csv', delimiter=';')
        print(f"Dataset inicial shape: {data.shape}")

        # Remove undefined behaviors
        # data = self.data_cleaner.remove_instances_with_value(data, self.target_column, '?')
        # exibindo a quantidade de classes em comportamento
        # print(f"Classes de comportamento: {data['comportamento'].unique()}")

        # Cria id único de sequencias
        data['sequence_id'] = self._create_sequence_ids(data)

        # Remove unnecessary columns usando configuração
        cleaned_data = self.data_cleaner.remove_columns(
            data, use_config=True)
        
        # Substitui comportamentos on-task-resource (chamado de on task out no algoritmo) e on-task-conversation por on-task-out
        cleaned_data[self.target_column] = cleaned_data[self.target_column].replace(
            ['ON TASK OUT', 'ON TASK CONVERSATION'], 'ON TASK OUT')
        # exibindo a quantidade de classes em comportamento
        print(
            f"Classes de comportamento: {cleaned_data[self.target_column].unique()}")


        return cleaned_data
    
    def _create_sequence_ids(self, X: pd.DataFrame) -> np.ndarray:
        return (X['aluno'].astype(int) * 10000 +
                X['num_dia'].astype(int) * 1000 +
                X['num_log'].astype(int))

    def prepare_data(self, data):
        """
        Prepara os dados para treinamento

        Note:
            - Mantém 'aluno' e 'comportamento' originais até após o split
            - Split é estratificado por aluno E comportamento
            - Remoção de colunas e transformações só após o split
        """

        print("\nIniciando preparação dos dados...")
        print(f"Dataset inicial - Shape: {data.shape}")
        print("Tipos de dados:")
        print(data.dtypes.value_counts())

        # Validação inicial das colunas necessárias
        self._validate_split_columns(data)

        # 1. Criar features temporais [NOVO]
        print("\nCriando features temporais...")
        temporal_processor = TemporalFeaturesProcessor()
        data = temporal_processor.fit_transform(data)
        print(f"Shape após features temporais: {data.shape}")

    
        # 2. Encode target (generally, not needed for most models)
        y = data[self.target_column]
        
        y_encoder = DataEncoder(
            categorical_threshold=5,
            scaling_strategy='standard',
            select_numerical=True,
            select_nominal=True,
            select_ordinal=False
        )        
        y_encoded = y_encoder.fit_transform_y(y)
        data[self.target_column] = y_encoded

        print(
            f"Distribuição original das classes:\n{data[self.target_column].value_counts()}")

        # 3. Divide the data into train and test sets stratified by student ID and target

        # Split estratificado usando novo identificador
        train_data, test_data = DataSplitter.split_stratified_by_groups(
            data=data,
            test_size=self.test_size,
            group_column='aluno',
            target_column=self.target_column
        )

        print(
            f"Distribuição no conjunto de treino:\n{train_data[self.target_column].value_counts()}")
        print(
            f"Distribuição no conjunto de teste:\n{test_data[self.target_column].value_counts()}")

        # Verifica a qualidade do split aqui, logo após a divisão
        self._verify_split_quality(train_data, test_data)


        # 5. Split features and target
        X_train, y_train = DataSplitter.split_into_x_y(
            train_data, self.target_column)
        X_test, y_test = DataSplitter.split_into_x_y(
            test_data, self.target_column)

        # 6. Impute missing values
        print("\nRealizando imputação de valores faltantes...")
        imputer = DataImputer(
            numerical_strategy='knn',
            categorical_strategy='most_frequent',
            knn_neighbors=5
        )

        # Importante: fit apenas no treino, transform em ambos
        print("Ajustando imputador nos dados de treino...")
        X_train_imputed = imputer.fit_transform(X_train)
        print("Aplicando imputação nos dados de teste...")
        X_test_imputed = imputer.transform(X_test)

        # 6. Encode features
        print("\nRealizando encoding das features...")
        X_encoder = DataEncoder(
            categorical_threshold=5,
            scaling_strategy='standard',
            select_numerical=True,
            select_nominal=True,
            select_ordinal=False
        )
        X_encoder.fit(X_train)
        X_train_encoded = X_encoder.transform(X_train_imputed)
        X_test_encoded = X_encoder.transform(X_test_imputed)

        logger = FeatureMappingLogger()
        logger.log_feature_mappings(X_encoder)
        logger.log_target_mappings(y_encoder)

        # Após todas as transformações
        print("\nResumo final do pré-processamento:")
        print(
            f"Shape final - X_train: {X_train_encoded.shape}, X_test: {X_test_encoded.shape}")

        return X_train_encoded, X_test_encoded, y_train, y_test

    def _log_dataset_changes(self, stage: str, data: pd.DataFrame):
        """Registra mudanças no dataset"""
        print(f"\nDataset {stage} - Shape: {data.shape}")
        print("Tipos de dados:")
        print(data.dtypes.value_counts())

    def _log_removed_items(self, item_type: str, items: list):
        """Registra itens removidos"""
        if items:
            print(f"\nRemovidos {len(items)} {item_type}:")
            print(items)

    def _log_class_distribution(self, stage: str, y: pd.Series):
        """Registra distribuição das classes"""
        print(f"\nDistribuição no conjunto de {stage}:")
        dist = y.value_counts()
        print(dist)
        print("\nPorcentagens:")
        print((dist/len(y)*100).round(2))

    def _validate_split_columns(self, data):
        """
        Valida se todas as colunas necessárias existem antes do split.

        Args:
            data (pd.DataFrame): Dados a serem validados

        Raises:
            ValueError: Se alguma coluna necessária estiver faltando
            ValueError: Se houver valores nulos nas colunas críticas
        """
        # Verifica presença das colunas
        required_columns = ['aluno', self.target_column]
        missing_columns = [
            col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(
                f"Colunas necessárias ausentes: {missing_columns}")

        # Verifica valores nulos em colunas críticas
        null_counts = data[required_columns].isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]

        if not columns_with_nulls.empty:
            raise ValueError(
                f"Valores nulos encontrados em colunas críticas:\n{columns_with_nulls}")

    def run(self):
        """Executa o pipeline completo de detecção de comportamentos."""
        print("Iniciando pipeline de detecção de comportamentos...")

        # Load and clean data
        print("\n1. Carregando e limpando dados...")
        data = self.load_and_clean_data()

        print("Inspecionando os dados após limpeza:")
        print("Dados de treino:")
        data.info()
        print("\nDescriptive Statistics:")
        print(data.describe(include='all'))

        # Prepare data
        print("\n2. Preparando dados para treinamento...")
        X_train, X_test, y_train, y_test = self.prepare_data(data)

        print("Inspecionando os dados após limpeza e pre-processamento:")
        print("Dados de treino:")
        X_train.info()
        print("\nDescriptive Statistics:")
        print(X_train.describe(include='all'))

        # Balance data
        print("\n3. Balanceando dados de treino...")
        X_train, y_train = self.balance_data(X_train, y_train, strategy=0.75)

        print(f"Modelos disponíveis: {self.model_params.get_models().keys()}")

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
            n_jobs=self.n_jobs
        )

        # Define training stages
        stages = self._get_training_stages()

        training_manager.execute_all_stages(stages)

        print("\nPipeline concluído!")
