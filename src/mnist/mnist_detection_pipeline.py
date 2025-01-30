from sklearn.datasets import load_digits
import pandas as pd

from core.pipeline.base_pipeline import BasePipeline
from core.preprocessors.data_splitter import DataSplitter
from core.preprocessors.data_encoder import DataEncoder
from mnist.digits_model_params import DigitsModelParams
from core.management.stage_training_manager import StageTrainingManager


class MNISTDetectionPipeline(BasePipeline):
    def __init__(self, target_column='target', n_iter=50, n_jobs=6, test_size=0.2):
        super().__init__(target_column=target_column,
                         n_iter=n_iter, n_jobs=n_jobs, test_size=test_size)

    def _get_model_params(self):
        """Implementação do método abstrato para obter parâmetros do modelo."""
        return DigitsModelParams()

    def load_and_clean_data(self):
        """Implementação do método abstrato para carregar e limpar dados."""
        # Carregar dataset MNIST
        print("Carregando dataset MNIST...")
        digits = load_digits()
        data = pd.DataFrame(digits.data)
        data['target'] = digits.target

        print(f"Dataset shape: {data.shape}")
        return data

    def prepare_data(self, data):
        print("Preparando dados...")
        # Split data
        print("Dividindo dados em treino e teste...")
        train_data, test_data = DataSplitter.split_data_stratified(
            data, test_size=self.test_size, target_column='target'
        )

        # Split features and target
        print("Separando features e target...")
        X_train, y_train = DataSplitter.split_into_x_y(train_data, 'target')
        X_test, y_test = DataSplitter.split_into_x_y(test_data, 'target')

        # Scale features
        print("Normalizando features...")
        encoder = DataEncoder(
            categorical_threshold=10,
            scaling_strategy='standard',
            select_numerical=True,
            select_nominal=False,  # não selecionar colunas nominais
            select_ordinal=False   # não selecionar colunas ordinais
        )
        X_train = encoder.fit_transform(X_train)
        X_test = encoder.transform(X_test)

        return X_train, X_test, y_train, y_test

    def run(self):
        print("Iniciando pipeline de detecção MNIST...")

        # Load and prepare data
        print("\n1. Carregando e preparando dados...")
        data = self.load_and_clean_data()

        # Prepare data
        print("\n2. Preparando dados para treinamento...")
        X_train, X_test, y_train, y_test = self.prepare_data(data)

        # Balance data
        print("\n3. Balanceando dados de treino...")
        # X_train, y_train = self.balance_data(X_train, y_train, strategy='auto')

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

        # Execute training stages
        training_manager.execute_all_stages(stages)

        print("\nPipeline concluído!")
