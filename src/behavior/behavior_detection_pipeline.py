"""
Copyright (c) 2025 Patricia Jaques
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License. To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
"""

import numpy as np
import pandas as pd

from behavior.behavior_feature_engineer import BehaviorFeatureEngineer
from behavior.temporal_features_processor import TemporalFeaturesProcessor
from core.preprocessors.data_encoder import DataEncoder
from core.preprocessors.data_imputer import DataImputer
from core.preprocessors.data_loader import DataLoader
from core.preprocessors.data_splitter import DataSplitter
from behavior.behavior_model_params import BehaviorModelParams
from core.pipeline.base_pipeline import BasePipeline
from behavior.behavior_model_params import BehaviorModelParams


class BehaviorDetectionPipeline(BasePipeline):
    def __init__(self, target_column='comportamento', n_iter=50, n_jobs=6,
                 val_size=0.25, test_size=0.2, group_feature='aluno',
                 training_strategy_name='optuna', use_voting_classifier=True):
        """
        Inicializa o pipeline de detecção de comportamentos.

        Args:
            target_column (str): Nome da coluna alvo
            n_iter (int): Número de iterações para otimização de hiperparâmetros
            n_jobs (int): Número de jobs paralelos para processamento
            test_size (float): Proporção dos dados para conjunto de teste
        """
        super().__init__(
            target_column=target_column,
            n_iter=n_iter,
            n_jobs=n_jobs,
            val_size=val_size, test_size=test_size,
            group_feature=group_feature,
            training_strategy_name=training_strategy_name,
            use_voting_classifier=use_voting_classifier
        )

    def _get_model_params(self):
        """Obtém os parâmetros do modelo de comportamento."""
        return BehaviorModelParams()

    def load_data(self):
        """Carrega o dataset."""
        data = DataLoader.load_data(
            self.paths['data'] / 'new_logs_labels.csv', delimiter=';')
        self.logger.info(f"Dataset inicial shape: {data.shape}")
        return data

    def clean_data(self, data):
        """Limpa o dataset."""
        # Cria id único de sequencias
        # data['sequence_id'] = self._create_sequence_ids(data)

        # 1. Fazer engenharia de features
        self.logger.info("\nRealizando engenharia de features...")
        feature_engineer = BehaviorFeatureEngineer()
        data = feature_engineer.transform(data)

        # Remove unnecessary columns
        columns_to_keep = ['aluno', 'num_dia',
                           'num_log', 'sequence_id', 'comportamento']
        columns_to_remove = self.data_cleaner.get_columns_to_remove(
            self.config_manager)
        cleaned_data = self.data_cleaner.clean_data(data, target_column=self.target_column, undefined_value='?',
                                                    columns_to_remove=columns_to_remove, columns_to_keep=columns_to_keep, handle_multicollinearity=True)

        # Substitui comportamentos on-task-resource (chamado de on task out no algoritmo) e on-task-conversation por on-task-out
        cleaned_data[self.target_column] = cleaned_data[self.target_column].replace(
            ['ON TASK OUT', 'ON TASK', 'ON SYSTEM', 'ON TASK CONVERSATION'], 'ON TASK')
        # exibindo a quantidade de classes em comportamento
        self.logger.info(
            f"Classes de comportamento: {cleaned_data[self.target_column].unique()}")

        return cleaned_data

    def _create_sequence_ids(self, X: pd.DataFrame) -> np.ndarray:
        return (
            # X['aluno'].astype(int) * 10000 +
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

        self.logger.info("\nIniciando preparação dos dados...")
        self.logger.info(f"Dataset inicial - Shape: {data.shape}")
        self.logger.info("Tipos de dados:")
        self.logger.info(data.dtypes.value_counts())

        # Validação inicial das colunas necessárias
        self._validate_split_columns(data)

        # 1. Criar features temporais [NOVO]
        # self.logger.info("\nCriando features temporais...")
        # temporal_processor = TemporalFeaturesProcessor()
        # data = temporal_processor.fit_transform(data)
        # self.logger.info(f"Shape após features temporais: {data.shape}")

        # 2. Encode target (generally, not needed for most models)
        y = data[self.target_column]

        self.encoder = DataEncoder(
            categorical_threshold=10,
            scaling_strategy='none',
            select_numerical=False,
            select_nominal=True,
            select_ordinal=False,
            target_column=self.target_column
        )
        y_encoded = self.encoder.fit_transform_y(y)
        data[self.target_column] = y_encoded

        # 3. Divide the data into train, val and test sets stratified by student ID and target
        train_data, val_data, test_data = DataSplitter.split_stratified_by_groups_and_target(
            data=data,
            val_size=0.30,
            test_size=self.test_size,
            group_column='aluno',
            target_column=self.target_column
        )

        # Verifica a qualidade do split aqui, logo após a divisão
        self._verify_split_distribution(
            data=data,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data
        )

        # # Removing column 'aluno' and 'num_dia' after strratified split (they are necessary for the split).
        # # I'm suspecting that the model is learning the student ID and the day number.
        # train_data = self.data_cleaner.remove_columns(
        #     train_data, columns_to_remove=['aluno', 'num_dia'])
        # val_data = self.data_cleaner.remove_columns(
        #     val_data, columns_to_remove=['aluno', 'num_dia'])
        # test_data = self.data_cleaner.remove_columns(
        #     test_data, columns_to_remove=['aluno', 'num_dia'])

        # 4. Divide the data into features and target
        X_train, y_train = DataSplitter.separate_features_and_target(
            train_data, self.target_column)
        X_val, y_val = DataSplitter.separate_features_and_target(
            val_data, self.target_column)
        X_test, y_test = DataSplitter.separate_features_and_target(
            test_data, self.target_column)

        self.logger.info(
            f"No prepare_data: Distribuição de logs por aluno e comportamento:\n{train_data.groupby(['aluno', self.target_column]).size()}")
        self.logger.info(
            f"No prepare_data: Número mínimo de logs por aluno: {train_data.groupby(['aluno']).size().min()}")

        # 5. Impute missing values
        self.logger.info("\nRealizando imputação de valores faltantes...")
        imputer = DataImputer(
            numerical_strategy='knn',
            categorical_strategy='most_frequent',
            knn_neighbors=5
        )

        # Importante: fit apenas no treino, transform em ambos
        self.logger.info("Ajustando imputador nos dados de treino...")
        X_train_imputed = imputer.fit_transform(X_train)
        self.logger.info("Aplicando imputação nos dados de validação...")
        X_val_imputed = imputer.transform(X_val)
        self.logger.info("Aplicando imputação nos dados de teste...")
        X_test_imputed = imputer.transform(X_test)

        # 6. Encode features
        self.logger.info("\nRealizando encoding das features...")
        self.encoder.fit(X_train)
        X_train_encoded = self.encoder.transform(X_train_imputed)
        X_val_encoded = self.encoder.transform(X_val_imputed)
        X_test_encoded = self.encoder.transform(X_test_imputed)

        return X_train_encoded, X_val_encoded, X_test_encoded, y_train, y_val, y_test

    def _log_dataset_changes(self, stage: str, data: pd.DataFrame):
        """Registra mudanças no dataset"""
        self.logger.info(f"\nDataset {stage} - Shape: {data.shape}")
        self.logger.info("Tipos de dados:")
        self.logger.info(data.dtypes.value_counts())

    def _log_removed_items(self, item_type: str, items: list):
        """Registra itens removidos"""
        if items:
            self.logger.info(f"\nRemovidos {len(items)} {item_type}:")
            self.logger.info(items)

    def _log_class_distribution(self, stage: str, y: pd.Series):
        """Registra distribuição das classes"""
        self.logger.info(f"\nDistribuição no conjunto de {stage}:")
        dist = y.value_counts()
        self.logger.info(dist)
        self.logger.info("\nPorcentagens:")
        self.logger.info((dist/len(y)*100).round(2))

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

    def _verify_split_distribution(self, data, train_data, val_data=None, test_data=None):
        """Verifica a qualidade do split"""
        self.logger.info("\nDistribuição de classes:")
        self.logger.info("\nOriginal:")
        self.logger.info(data[self.target_column].value_counts(normalize=True))

        self.logger.info("\nTreino:")
        self.logger.info(
            train_data[self.target_column].value_counts(normalize=True))

        if val_data is not None:
            self.logger.info("\nValidação:")
            self.logger.info(
                val_data[self.target_column].value_counts(normalize=True))

        self.logger.info("\nTeste:")
        self.logger.info(
            test_data[self.target_column].value_counts(normalize=True))

        if self.group_feature is not None:
            # Verifica sobreposição de grupos
            train_groups = set(train_data[self.group_feature].unique())
            test_groups = set(test_data[self.group_feature].unique())
            overlap = train_groups.intersection(test_groups)

            if len(overlap) > 0:
                self.logger.info(
                    f"\nALERTA: Existem {len(overlap)} grupos sobrepostos entre treino e teste!")

            if val_data is not None:
                val_groups = set(val_data[self.group_feature].unique())
                overlap_val_train = val_groups.intersection(train_groups)
                overlap_val_test = val_groups.intersection(test_groups)

                if len(overlap_val_train) > 0 or len(overlap_val_test) > 0:
                    self.logger.info(
                        f"\nALERTA: Existem grupos sobrepostos na validação!")
        else:
            self.logger.info(
                "\ngroup_feature is None, pulando verificações de grupo.")
