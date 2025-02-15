import os
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from core.preprocessors.data_cleaner import DataCleaner
from core.config.config_manager import ConfigManager
from core.preprocessors.data_loader import DataLoader
from core.reporting.classification_model_metrics import ClassificationModelMetrics
from core.reporting.metrics_reporter import MetricsReporter
from core.utils.path_manager import PathManager
from core.preprocessors.data_encoder import DataEncoder
from core.lstm.lstm_pipeline import LSTMPipeline
from core.lstm.lstm_dataset import LSTMDataset


class LSTMBehaviorPipeline:
    def __init__(self, data_path, module_name, target_column, sequence_length=10, test_size=0.2, random_state=42, lstm_params=None):
        self.data_path = data_path
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.random_state = random_state
        self.module_name = module_name
        PathManager.set_module(self.module_name)
        self.config_manager = ConfigManager()
        self.data_cleaner = DataCleaner()
        self.encoder = DataEncoder(select_numerical=True, select_nominal=True, select_ordinal=False)
        self.lstm_params = lstm_params if lstm_params is not None else {
            'hidden_size': 128,
            'num_layers': 2,
            'batch_size': 32,
            'num_epochs': 20,
            'learning_rate': 0.001
        }

    def load_data(self):
        print("Loading data...")
        data = DataLoader.load_data(self.data_path, delimiter=';')
        return data

    def clean_data(self, data):
        data = data[data[self.target_column].notna()]
        print(f"Unique classes before cleaning: {data[self.target_column].unique()}")

        columns_to_keep = ['id_log', 'aluno', 'grupo', 'num_dia', 'num_log', 'log_type', self.target_column]
        columns_to_remove = self.data_cleaner.get_columns_to_remove(self.config_manager)

        cleaned_data = self.data_cleaner.clean_data(
            data,
            target_column=self.target_column,
            undefined_value='?',
            columns_to_remove=columns_to_remove,
            columns_to_keep=columns_to_keep,
            handle_multicollinearity=True
        )

        print(f"Data shape after cleaning: {cleaned_data.shape}")
        print(f"Unique classes after cleaning: {cleaned_data[self.target_column].unique()}")
        return cleaned_data

    def _create_sequence_ids(self, X: pd.DataFrame) -> np.ndarray:
        return (
            X['aluno'].astype(int) * 10000 +
            X['num_dia'].astype(int) * 1000 +
            X['num_log'].astype(int))

    def encode_data(self, cleaned_data):
        features = cleaned_data.drop(columns=[self.target_column])
        target = cleaned_data[self.target_column]
        print("Applying one-hot encoding to  features...")
        encoded_features = self.encoder.fit_transform(features)
        encoded_data = pd.concat([encoded_features, target], axis=1)
        return encoded_data

    def prepare_data(self, data):
        print("\nDiagnosticando dados antes de prepare_sequences:")
        print(f"Shape dos dados: {data.shape}")
        print(f"Colunas disponíveis: {data.columns.tolist()}")
        print(f"Valores únicos no target: {data[self.target_column].unique()}")
        print(
            f"Quantidade de registros por classe:\n{data[self.target_column].value_counts()}")

        if len(data) < self.sequence_length:
            raise ValueError(
                f"Dados insuficientes para criar sequências de tamanho {self.sequence_length}")

        sequences, labels = LSTMDataset.prepare_sequences(
            df=data,
            sequence_length=self.sequence_length,
            feature_cols=None,
            target_column=self.target_column,
            data_encoder=self.encoder,  # Usando o DataEncoder ao invés do LabelEncoder
            group_by_col='aluno',
            sort_by_cols=['num_dia', 'num_log']
        )

        if sequences is not None:
            print(f"\nShape das sequences: {sequences.shape}")
        if labels is not None:
            print(f"Shape dos labels: {labels.shape}")

        return sequences, labels

    def run(self):
        # Load and clean data
        data = self.load_data()
        cleaned_data = self.clean_data(data)

        # Initialize the DataEncoder
        self.encoder.fit(cleaned_data.drop(columns=[self.target_column]))

        # Prepare sequences
        sequences, labels = self.prepare_data(cleaned_data)

        if sequences is None or labels is None:
            raise ValueError("Error: sequences or labels are None.")

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels
        )

        # Create and train the model
        lstm_pipeline = LSTMPipeline(
            target_column=self.target_column,
            hidden_size=self.lstm_params['hidden_size'],
            num_layers=self.lstm_params['num_layers'],
            sequence_length=self.sequence_length,
            batch_size=self.lstm_params['batch_size'],
            num_epochs=self.lstm_params['num_epochs'],
            learning_rate=self.lstm_params['learning_rate'],
            data_encoder=self.encoder
        )

        print("\nTraining LSTM model...")
        lstm_pipeline.fit(X_train, y_train)

        print("\nEvaluating LSTM model...")
        train_pred = lstm_pipeline.predict(X_train, return_encoded=True)
        test_pred = lstm_pipeline.predict(X_test, return_encoded=True)

        # Gerar métricas para treino e teste
        self._generate_metrics_report(y_train, y_test, lstm_pipeline, train_pred, test_pred)

        return lstm_pipeline

    def _generate_metrics_report(self, y_train, y_test, lstm_pipeline, train_pred, test_pred):
        metrics = ClassificationModelMetrics(
            stage_name="lstm_model",
            train_metrics={
                'balanced_accuracy': balanced_accuracy_score(y_train, train_pred),
                'precision': precision_score(y_train, train_pred, average='weighted'),
                'recall': recall_score(y_train, train_pred, average='weighted'),
                'f1-score': f1_score(y_train, train_pred, average='weighted'),
                'kappa': cohen_kappa_score(y_train, train_pred)
            },
            test_metrics={
                'balanced_accuracy': balanced_accuracy_score(y_test, test_pred),
                'precision': precision_score(y_test, test_pred, average='weighted'),
                'recall': recall_score(y_test, test_pred, average='weighted'),
                'f1-score': f1_score(y_test, test_pred, average='weighted'),
                'kappa': cohen_kappa_score(y_test, test_pred)
            },
            val_metrics=None,  # Não temos conjunto de validação neste momento
            class_report_train=classification_report(
                y_train, train_pred, output_dict=True),
            class_report_test=classification_report(
                y_test, test_pred, output_dict=True),
            class_report_val=None,
            confusion_matrix_train=(confusion_matrix(y_train, train_pred),),
            confusion_matrix_test=(confusion_matrix(y_test, test_pred),),
            confusion_matrix_val=None,
            class_labels=lstm_pipeline.get_classes(),
            training_type="LSTM",
            hyperparameters=self.lstm_params,
            execution_time=lstm_pipeline.trainer.execution_time if hasattr(
                lstm_pipeline.trainer, 'execution_time') else None
        )

        # Gerar relatórios usando o MetricsReporter
        metrics_reporter = MetricsReporter()
        metrics_reporter.generate_stage_report(metrics)

        # Imprimir métricas de teste no console também
        print("\nClassification Report:")
        print(classification_report(
            y_test,
            test_pred,
            target_names=lstm_pipeline.get_classes()
        ))
        print(
            f"\nBalanced Accuracy: {balanced_accuracy_score(y_test, test_pred):.4f}")


def main():
    DATA_PATH = '/Users/patricia/Documents/code/python-code/behavior-detection/data/new_logs_labels.csv'
    TARGET_COLUMN = 'estado_afetivo'
    SEQUENCE_LENGTH = 10

    pipeline = LSTMBehaviorPipeline(
        data_path=DATA_PATH,
        module_name='emotion',
        target_column=TARGET_COLUMN,
        sequence_length=SEQUENCE_LENGTH
    )
    pipeline.run()


if __name__ == "__main__":
    os.system('clear')
    main()