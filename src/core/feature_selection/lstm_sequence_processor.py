from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import StandardScaler
import torch
from behavior.lstm_behavior_model import BehaviorSequenceDataset, LSTMBehaviorModel
from core.preprocessors.data_loader import DataLoader


class LSTMSequenceProcessor(BaseEstimator, TransformerMixin):
    """
    Processa sequências de comportamentos para o modelo LSTM, integrado com scikit-learn.
    """

    def __init__(self,
                 sequence_length: int = 10,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 batch_size: int = 32,
                 num_epochs: int = 10,
                 learning_rate: float = 0.001,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.scaler = StandardScaler()
        self.model = None

    def _create_sequences(self, X: pd.DataFrame) -> np.ndarray:
        """Cria sequências de características para cada aluno."""
        sequences = []
        students = X['aluno'].unique()

        for student in students:
            student_data = X[X['aluno'] == student].sort_values(
                ['num_dia', 'num_log'])
            student_features = student_data.drop(
                ['aluno', 'num_dia', 'num_log'], axis=1)

            # Padding/truncating para garantir mesmo tamanho
            if len(student_features) < self.sequence_length:
                pad_length = self.sequence_length - len(student_features)
                padding = np.zeros((pad_length, student_features.shape[1]))
                student_sequence = np.vstack([padding, student_features])
            else:
                student_sequence = student_features.iloc[-self.sequence_length:].values

            sequences.append(student_sequence)

        return np.array(sequences)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        # Criar sequências
        sequences = self._create_sequences(X)

        if y is not None:
            # Mapear labels para mesma ordem das sequências
            labels = []
            for student in X['aluno'].unique():
                student_labels = y[X['aluno'] == student].iloc[-1]
                labels.append(student_labels)

            self.num_classes = len(np.unique(labels))

            # Criar e treinar modelo
            input_size = sequences.shape[2]
            self.model = LSTMBehaviorModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_classes=self.num_classes
            ).to(self.device)

            # Preparar dataset
            dataset = BehaviorSequenceDataset(sequences, labels)
            dataloader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True)

            # Treinar
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate)

            self.model.train()
            for epoch in range(self.num_epochs):
                for batch_sequences, batch_labels in dataloader:
                    batch_sequences = batch_sequences.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_sequences)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforma dados em sequências e faz previsões se modelo estiver treinado."""
        sequences = self._create_sequences(X)

        if self.model is not None:
            self.model.eval()
            with torch.no_grad():
                sequences_tensor = torch.FloatTensor(sequences).to(self.device)
                outputs = self.model(sequences_tensor)
                predictions = outputs.argmax(dim=1).cpu().numpy()
            return predictions

        return sequences

    def get_params(self, deep=True):
        return {
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
