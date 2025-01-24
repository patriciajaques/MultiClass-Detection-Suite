import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import LabelEncoder
from .lstm_dataset import LSTMDataset
from .lstm_trainer import LSTMTrainer


class LSTMPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_size=64, num_layers=2, sequence_length=10,
                 batch_size=32, num_epochs=10, learning_rate=0.001):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.trainer = None
        self.label_encoder = LabelEncoder()

    def fit(self, X, y):
        # Fit do label encoder com as labels originais
        self.label_encoder.fit(y)
        print(f"Classes identificadas: {self.label_encoder.classes_}")
        print(f"Mapeamento de classes: {dict(zip(self.label_encoder.classes_,self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        # Garantir que a coluna comportamento esteja presente no DataFrame
        X_with_labels = X.copy()
        X_with_labels['comportamento'] = y
        
        # Preparar as sequências
        sequences, labels = LSTMDataset.prepare_sequences(
            X_with_labels,
            sequence_length=self.sequence_length,
            behavior_labels=y,
            label_encoder=self.label_encoder
        )
        
        print(f"Shape das sequências: {sequences.shape}")
        print(f"Shape das labels: {labels.shape}")
        
        # Criar dataset para treinamento
        train_dataset = LSTMDataset(sequences, labels)
        
        input_size = sequences.shape[2]
        num_classes = len(self.label_encoder.classes_)
        
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
        return self
    

    def predict(self, X):
        """
        Realiza predições para novos dados.
        
        Args:
            X: DataFrame com os dados de input
        Returns:
            np.ndarray: Array com as predições
        """
        print("\nRealizando predições:")
        print(f"Shape dos dados de entrada: {X.shape}")

        # Preparar as sequências para predição
        X_sequences, _ = LSTMDataset.prepare_sequences(
            df=X,
            sequence_length=self.sequence_length,
            behavior_labels=None,
            label_encoder=None
        )

        print(f"Shape das sequências preparadas: {X_sequences.shape}")

        # Criar dataset para predição
        test_dataset = LSTMDataset(X_sequences)

        # Obter predições numéricas do modelo
        numeric_predictions = self.trainer.predict(test_dataset)

        # Converter predições numéricas de volta para labels originais
        predictions = self.label_encoder.inverse_transform(numeric_predictions)

        print(f"Número de predições: {len(predictions)}")
        print(f"Classes únicas preditas: {np.unique(predictions)}")

        return predictions

    def predict_proba(self, X):
        """
        Retorna probabilidades para cada classe.
        
        Args:
            X: DataFrame com os dados de input
        Returns:
            np.ndarray: Array com probabilidades para cada classe
        """
        if not hasattr(self.trainer, 'predict_proba'):
            raise NotImplementedError(
                "predict_proba não está implementado no trainer")

        # Preparar as sequências
        X_sequences, _ = LSTMDataset.prepare_sequences(
            df=X,
            sequence_length=self.sequence_length,
            behavior_labels=None,
            label_encoder=None
        )

        # Criar dataset
        test_dataset = LSTMDataset(X_sequences)

        # Obter probabilidades
        return self.trainer.predict_proba(test_dataset)
