from sklearn.datasets import load_digits
import pandas as pd

from core.pipeline.base_pipeline import BasePipeline
from core.preprocessors.data_splitter import DataSplitter
from core.preprocessors.data_encoder import DataEncoder
from core.preprocessors.data_balancer import DataBalancer
from core.models.multiclass.digits_model_params import DigitsModelParams
from core.management.stage_training_manager import StageTrainingManager


class MNISTDetectionPipeline(BasePipeline):
    def __init__(self, n_iter=50, n_jobs=6, test_size=0.2, base_path=None, stage_range=None):
        super().__init__(n_iter=n_iter, n_jobs=n_jobs, test_size=test_size,
                         base_path=base_path, stage_range=stage_range)

    def _get_model_params(self):
        """Implementação do método abstrato para obter parâmetros do modelo."""
        return DigitsModelParams()

    def load_and_clean_data(self):
        """Implementação do método abstrato para carregar e limpar dados."""
        return self.load_and_prepare_data()  # Reusa o método existente

    def load_and_prepare_data(self):
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
            num_classes=10,
            scaling_strategy='standard',
            select_numerical=True,
            select_nominal=False,  # Modificação aqui: não selecionar colunas nominais
            select_ordinal=False   # Modificação aqui: não selecionar colunas ordinais
        )
        X_train = encoder.fit_transform(X_train)
        X_test = encoder.transform(X_test)

        return X_train, X_test, y_train, y_test

    def balance_data(self, X_train, y_train):
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
        X_resampled, y_resampled = data_balancer.apply_smote(X_train, y_train)

        print(f"Shape de X_train após balanceamento: {X_resampled.shape}")
        print(f"Distribuição de classes após balanceamento:")
        print(pd.Series(y_resampled).value_counts())

        return X_resampled, y_resampled

    def run(self):
        print("Iniciando pipeline de detecção MNIST...")

        # Load and prepare data
        print("\n1. Carregando e preparando dados...")
        data = self.load_and_prepare_data()

        # Prepare data
        print("\n2. Preparando dados para treinamento...")
        X_train, X_test, y_train, y_test = self.prepare_data(data)

        # Balance data
        print("\n3. Balanceando dados de treino...")
        X_train, y_train = self.balance_data(X_train, y_train)

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
            n_jobs=self.n_jobs,
            stage_range=self.stage_range
        )

        # Define training stages
        stages = self._get_training_stages()

        # Execute training stages
        print(
            f"\n5. Executando stages {self.stage_range if self.stage_range else 'todos'}...")
        training_manager.execute_all_stages(training_manager, stages)

        print("\nPipeline concluído!")
