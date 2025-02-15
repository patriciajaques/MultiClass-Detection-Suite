import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score

from core.evaluation.model_evaluator import ModelEvaluator
from core.reporting.classification_model_metrics import ClassificationModelMetrics
from core.reporting.metrics_reporter import MetricsReporter
from .lstm_dataset import LSTMDataset
from .lstm_trainer import LSTMTrainer


class LSTMPipeline(BaseEstimator, ClassifierMixin):
    """
    Pipeline for LSTM-based sequence classification.
    Implements scikit-learn's estimator interface.
    """

    def __init__(self, target_column='target', hidden_size=64, num_layers=2, sequence_length=10,
                 batch_size=32, num_epochs=10, learning_rate=0.001, data_encoder=None):
        """
        Initialize the LSTM Pipeline.

        Args:
            target_column (str): Name of the target column
            hidden_size (int): Number of features in the hidden state
            num_layers (int): Number of LSTM layers
            sequence_length (int): Length of input sequences
            batch_size (int): Size of batches for training
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate for optimization
            data_encoder: Custom encoder for feature and target transformation
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.trainer = None
        self.target_column = target_column
        self.data_encoder = data_encoder
        self.metrics_reporter = MetricsReporter()

    def fit(self, X, y):
        """
        Train the LSTM model.
        
        Args:
            X: numpy array of shape (n_samples, sequence_length, n_features)
               containing the prepared sequences
            y: numpy array of shape (n_samples,) containing the encoded labels
        
        Returns:
            self: Returns the instance itself
        """
        print(f"Input sequences shape: {X.shape}")
        print(f"Input labels shape: {y.shape}")

        # Create training dataset
        train_dataset = LSTMDataset(X, y)

        input_size = X.shape[2]  # number of features
        num_classes = len(np.unique(y))

        print(f"Input size: {input_size}")
        print(f"Number of classes: {num_classes}")

        self.trainer = LSTMTrainer(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=num_classes,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate
        )

        self.trainer.train(train_dataset)

        # Generate train metrics report
        train_pred = self.predict(X)
        metrics = self._generate_metrics_report(
            y_true=y,
            y_pred=train_pred,
            training_history=self.trainer.training_history,
            execution_time=self.trainer.execution_time
        )

        # Generate and log the training report
        self.metrics_reporter.generate_stage_report(metrics)
        return self
        return self

    def predict(self, X, return_encoded=True):
        """
        Make predictions for new data.
        
        Args:
            X: numpy array of shape (n_samples, sequence_length, n_features)
               containing the prepared sequences
            return_encoded (bool): If True, returns encoded (numeric) predictions,
                                 if False, returns original string labels
        
        Returns:
            np.ndarray: Array containing the predicted labels
        """
        print("\nMaking predictions:")
        print(f"Input data shape: {X.shape}")

        # Create dataset for prediction
        test_dataset = LSTMDataset(X)

        # Get numeric predictions from the model
        numeric_predictions = self.trainer.predict(test_dataset)

        if not return_encoded:
            # Convert numeric predictions back to original labels only if requested
            predictions = self.data_encoder.inverse_transform_y(
                numeric_predictions)
            print(f"Unique predicted classes: {np.unique(predictions)}")
        else:
            predictions = numeric_predictions
            print(f"Unique predicted class indices: {np.unique(predictions)}")

        print(f"Number of predictions: {len(predictions)}")
        return predictions

    def predict_proba(self, X):
        """
        Return probability estimates for each class.
        
        Args:
            X: numpy array of shape (n_samples, sequence_length, n_features)
               containing the prepared sequences
        
        Returns:
            np.ndarray: Array containing probabilities for each class
        """
        if not hasattr(self.trainer, 'predict_proba'):
            raise NotImplementedError(
                "predict_proba is not implemented in the trainer")

        # Create dataset for prediction
        test_dataset = LSTMDataset(X)

        # Get probability estimates
        return self.trainer.predict_proba(test_dataset)

    def get_classes(self):
        """
        Get the list of class labels in order of their numeric encoding.
        
        Returns:
            list: List of class labels
        """
        if self.data_encoder is None:
            return None
        class_mapping = self.data_encoder.get_class_mapping()
        return [class_mapping[i] for i in range(len(class_mapping))]

    def _generate_metrics_report(self, y_true, y_pred, training_history, execution_time):
        """Gera relatório de métricas no formato esperado pelo sistema"""

        train_metrics = {
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1-score': f1_score(y_true, y_pred, average='weighted'),
            'kappa': cohen_kappa_score(y_true, y_pred)
        }

        metrics = ClassificationModelMetrics(
            stage_name="lstm_model",
            train_metrics=train_metrics,
            val_metrics=None,  # Não temos validação neste momento
            test_metrics=None,  # Não temos teste neste momento
            class_report_train=classification_report(
                y_true, y_pred, output_dict=True),
            class_report_val=None,
            class_report_test=None,
            confusion_matrix_train=(confusion_matrix(y_true, y_pred),),
            confusion_matrix_val=None,
            confusion_matrix_test=None,
            class_labels=list(np.unique(y_true)),
            training_type="LSTM",
            hyperparameters={
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs
            },
            execution_time=execution_time
        )

        return metrics
