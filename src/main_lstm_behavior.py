import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from core.logging.logger_config import LoggerConfig
from core.preprocessors.data_cleaner import DataCleaner
from core.config.config_manager import ConfigManager
from core.preprocessors.data_loader import DataLoader
from core.preprocessors.data_splitter import DataSplitter
from core.reporting.classification_model_metrics import ClassificationModelMetrics
from core.reporting.metrics_reporter import MetricsReporter
from core.utils.path_manager import PathManager
from core.preprocessors.data_encoder import DataEncoder
from core.lstm.lstm_pipeline import LSTMPipeline
from core.lstm.lstm_dataset import LSTMDataset


class LSTMBehaviorPipeline:
    def __init__(
        self,
        data_path,
        module_name,
        target_column="estado_afetivo",
        group_column=None,
        sequence_length=10,
        val_size=None,
        test_size=0.2,
        random_state=42,
        hidden_size=128,
        num_layers=2,
        batch_size=32,
        num_epochs=20,
        learning_rate=0.001,
    ):
        # Basic configuration
        self.data_path = data_path
        self.target_column = target_column
        self.group_column = group_column
        self.sequence_length = sequence_length
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.module_name = module_name

        # LSTM parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # Initialize required components
        PathManager.set_module(self.module_name)
        self.logger = LoggerConfig.get_logger("lstm_training")
        self.config_manager = ConfigManager()
        self.data_cleaner = DataCleaner()
        self.encoder = DataEncoder(
            categorical_threshold=6,  # Columns with <= 6 unique values treated as categorical
            scaling_strategy="minmax",  # MinMaxScaler for numerical features
            select_numerical=True,  # Enable numerical feature scaling
            select_nominal=True,  # Enable one-hot encoding for categorical features
            select_ordinal=False,  # Disable ordinal encoding
            target_column=target_column,
        )
        self.metrics_reporter = MetricsReporter()

    def load_data(self):
        print("Loading data...")
        data = DataLoader.load_data(self.data_path, delimiter=";")
        return data

    def clean_data(self, data):
        data = data[data[self.target_column].notna()]
        print(f"Unique classes before cleaning: {data[self.target_column].unique()}")

        columns_to_keep = [
            "id_log",
            "aluno",
            "grupo",
            "num_dia",
            "num_log",
            "log_type",
            self.target_column,
        ]
        columns_to_remove = self.data_cleaner.get_columns_to_remove(self.config_manager)

        cleaned_data = self.data_cleaner.clean_data(
            data,
            target_column=self.target_column,
            undefined_value="?",
            columns_to_remove=columns_to_remove,
            columns_to_keep=columns_to_keep,
            handle_multicollinearity=True,
        )

        print(f"Data shape after cleaning: {cleaned_data.shape}")
        print(
            f"Unique classes after cleaning: {cleaned_data[self.target_column].unique()}"
        )
        return cleaned_data

    def _create_sequence_ids(self, X: pd.DataFrame) -> np.ndarray:
        return (
            X["aluno"].astype(int) * 10000
            + X["num_dia"].astype(int) * 1000
            + X["num_log"].astype(int)
        )

    def encode_data(self, train_data, val_data, test_data):
        """
        Encodes training, validation, and test data using the new DataEncoder.

        If the target column is present:
        - When more than one column is provided, the encoder fits/transforms both features and target.
        - When only the target column is provided, only the target is encoded.

        The method returns a concatenated DataFrame of encoded features and target (if available)
        for each split.
        """
        # --- Encode Training Data ---
        if self.target_column in train_data.columns:
            # Fit encoder on training data (includes both features and target)
            train_enc = self.encoder.fit_transform(train_data)
            if isinstance(train_enc, tuple):
                train_features_enc, train_target_enc = train_enc
            else:
                train_features_enc = train_enc
                train_target_enc = None
        else:
            train_features_enc = self.encoder.fit_transform_features(train_data)
            train_target_enc = None

        # --- Encode Test Data ---
        if self.target_column in test_data.columns:
            test_enc = self.encoder.transform(test_data)
            if isinstance(test_enc, tuple):
                test_features_enc, test_target_enc = test_enc
            else:
                test_features_enc = test_enc
                test_target_enc = None
        else:
            test_features_enc = self.encoder.transform_features(test_data)
            test_target_enc = None

        # --- Encode Validation Data (if provided) ---
        if val_data is not None:
            if self.target_column in val_data.columns:
                val_enc = self.encoder.transform(val_data)
                if isinstance(val_enc, tuple):
                    val_features_enc, val_target_enc = val_enc
                else:
                    val_features_enc = val_enc
                    val_target_enc = None
            else:
                val_features_enc = self.encoder.transform_features(val_data)
                val_target_enc = None
        else:
            val_features_enc, val_target_enc = None, None

        # --- Concatenate encoded features and target (if target encoding exists) ---
        if train_target_enc is not None:
            # Convert target arrays to DataFrames (using the target column name)
            train_target_df = pd.DataFrame(train_target_enc, columns=[self.target_column])
            test_target_df = pd.DataFrame(test_target_enc, columns=[self.target_column])
            if val_target_enc is not None:
                val_target_df = pd.DataFrame(val_target_enc, columns=[self.target_column])
            else:
                val_target_df = None

            train_data_enc = pd.concat([train_features_enc, train_target_df], axis=1)
            test_data_enc = pd.concat([test_features_enc, test_target_df], axis=1)
            if val_features_enc is not None and val_target_df is not None:
                val_data_enc = pd.concat([val_features_enc, val_target_df], axis=1)
            else:
                val_data_enc = val_features_enc
        else:
            train_data_enc = train_features_enc
            test_data_enc = test_features_enc
            val_data_enc = val_features_enc

        return train_data_enc, val_data_enc, test_data_enc

    def prepare_data(self, cleaned_data):
        """
        Prepares data for LSTM training, ensuring proper stratification, encoding and sequence creation.

        The process follows these steps:
        1. Stratified split by group
        2. Fit and apply encoder only on training data (prevent data leakage)
        3. Verify split quality
        4. Create sequences for LSTM
        """
        self.logger.info("\nStarting data preparation...")

        # 1. Perform stratified split with elegant unpacking
        split_result = DataSplitter.split_stratified_by_groups(
            data=cleaned_data,
            group_column="aluno",
            test_size=self.test_size,
            val_size=self.val_size,
        )
        train_data, *middle, test_data = split_result
        val_data = middle[0] if middle else None

        # 2. Verify split quality
        split_metrics = DataSplitter.verify_split_quality(
            data=cleaned_data,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            target_column=self.target_column,
            group_column=self.group_column,
            logger=self.logger,
        )

        # 3. Handle encoding (preventing data leakage)
        train_data, val_data, test_data = self.encode_data(
            train_data, val_data, test_data
        )

        # 4. Create sequences for LSTM processing
        self.logger.info("Creating sequences for each dataset...")

        # Training sequences
        train_sequences, train_labels = LSTMDataset.prepare_sequences(
            df=train_data,
            sequence_length=self.sequence_length,
            target_column=self.target_column,
            group_by_col="aluno",
            sort_by_cols=["num_dia", "num_log"],
            data_encoder=None,  # Since data is already encoded
        )

        # Test sequences
        test_sequences, test_labels = LSTMDataset.prepare_sequences(
            df=test_data,
            sequence_length=self.sequence_length,
            target_column=self.target_column,
            group_by_col="aluno",
            sort_by_cols=["num_dia", "num_log"],
            data_encoder=None,
        )

        # Validation sequences (if validation set exists)
        if val_data is not None:
            val_sequences, val_labels = LSTMDataset.prepare_sequences(
                df=val_data,
                sequence_length=self.sequence_length,
                target_column=self.target_column,
                group_by_col="aluno",
                sort_by_cols=["num_dia", "num_log"],
                data_encoder=None,
            )
        else:
            val_sequences, val_labels = None, None

        # Log sequence shapes
        self.logger.info("\nSequence shapes:")
        self.logger.info(f"Training sequences: {train_sequences.shape}")
        if val_sequences is not None:
            self.logger.info(f"Validation sequences: {val_sequences.shape}")
        self.logger.info(f"Test sequences: {test_sequences.shape}")

        return (
            train_sequences,
            val_sequences,
            test_sequences,
            train_labels,
            val_labels,
            test_labels,
        )

    def train_model(self, sequences, labels):
        lstm_pipeline = LSTMPipeline(
            target_column=self.target_column,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            data_encoder=self.encoder,
        )
        lstm_pipeline.fit(sequences, labels)
        return lstm_pipeline

    def _generate_metrics_report(
            self, y_train_true, y_train_pred, y_test_true, y_test_pred, training_history, execution_time
        ):
            """
            Generates metrics report for LSTM model with both training and test metrics.
            
            Args:
                y_train_true: True labels for training data
                y_train_pred: Predicted labels for training data
                y_test_true: True labels for test data
                y_test_pred: Predicted labels for test data
                training_history: Dictionary containing training metrics history
                execution_time: Total execution time of the training process
                
            Returns:
                ClassificationModelMetrics object containing all computed metrics
            """
            try:
                # Calculate training metrics
                train_metrics = {
                    'balanced_accuracy': balanced_accuracy_score(y_train_true, y_train_pred),
                    'precision': precision_score(y_train_true, y_train_pred, average='weighted'),
                    'recall': recall_score(y_train_true, y_train_pred, average='weighted'),
                    'f1-score': f1_score(y_train_true, y_train_pred, average='weighted'),
                    'kappa': cohen_kappa_score(y_train_true, y_train_pred)
                }

                # Calculate test metrics
                test_metrics = {
                    'balanced_accuracy': balanced_accuracy_score(y_test_true, y_test_pred),
                    'precision': precision_score(y_test_true, y_test_pred, average='weighted'),
                    'recall': recall_score(y_test_true, y_test_pred, average='weighted'),
                    'f1-score': f1_score(y_test_true, y_test_pred, average='weighted'),
                    'kappa': cohen_kappa_score(y_test_true, y_test_pred)
                }

                # Create metrics object with correct test data
                metrics = ClassificationModelMetrics(
                    stage_name="lstm_model",
                    train_metrics=train_metrics,
                    val_metrics=None,
                    test_metrics=test_metrics,
                    class_report_train=classification_report(y_train_true, y_train_pred, output_dict=True),
                    class_report_val=None,
                    class_report_test=classification_report(y_test_true, y_test_pred, output_dict=True),  # CORRIGIDO
                    confusion_matrix_train=(confusion_matrix(y_train_true, y_train_pred),),
                    confusion_matrix_val=None,
                    confusion_matrix_test=(confusion_matrix(y_test_true, y_test_pred),),  # CORRIGIDO
                    class_labels=list(np.unique(y_train_true)),
                    training_type="LSTM",
                    hyperparameters={
                        "hidden_size": self.hidden_size,
                        "num_layers": self.num_layers,
                        "batch_size": self.batch_size,
                        "learning_rate": self.learning_rate,
                        "num_epochs": self.num_epochs,
                    },
                    execution_time=execution_time,
                )

                # Em _generate_metrics_report, adicione este log antes de retornar:
                self.logger.info(f"Train metrics: {train_metrics}")
                self.logger.info(f"Test metrics: {test_metrics}")

                return metrics
                
            except Exception as e:
                self.logger.error(f"Error generating metrics report: {str(e)}")
                self.logger.error("Full error traceback:", exc_info=True)
                return None

    def run(self):
        """Executa o pipeline LSTM."""
        print("Loading data...")
        data = self.load_data()

        print("Cleaning data...")
        cleaned_data = self.clean_data(data)

        print("Preparing sequences...")
        train_sequences, val_sequences, test_sequences, train_labels, val_labels, test_labels = self.prepare_data(cleaned_data)

        # Train model
        print("\nTraining LSTM model...")
        lstm_pipeline = self.train_model(train_sequences, train_labels)

        # Generate predictions
        train_pred = lstm_pipeline.predict(train_sequences)
        y_test_pred = lstm_pipeline.predict(test_sequences)

        # Generate metrics report
        metrics = self._generate_metrics_report(
            y_train_true=train_labels,
            y_train_pred=train_pred,
            y_test_true=test_labels,
            y_test_pred=y_test_pred,
            training_history=lstm_pipeline.trainer.training_history,
            execution_time=lstm_pipeline.trainer.execution_time,
        )

        # Generate reports
        self.metrics_reporter.generate_stage_report(metrics)

        return lstm_pipeline


def main():
    DATA_PATH = "/Users/patricia/Documents/code/python-code/behavior-detection/data/new_logs_labels.csv"
    TARGET_COLUMN = "estado_afetivo"
    SEQUENCE_LENGTH = 10

    pipeline = LSTMBehaviorPipeline(
        data_path=DATA_PATH,
        module_name="emotion",
        val_size=None,
        test_size=0.2,
        target_column=TARGET_COLUMN,
        group_column="aluno",
        sequence_length=SEQUENCE_LENGTH,
    )
    pipeline.run()


if __name__ == "__main__":
    os.system("clear")
    main()
