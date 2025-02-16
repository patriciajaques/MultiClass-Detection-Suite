import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from core.logging.logger_config import LoggerConfig
from .lstm_dataset import LSTMDataset
from .lstm_model import LSTMModel


class LSTMTrainer:
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=2,
                 num_classes=5,
                 learning_rate=0.001,
                 batch_size=32,
                 num_epochs=10,
                 device=None):

        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes
        ).to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'epoch_times': []
        }
        self.logger = LoggerConfig.get_logger('lstm_training')
        self.start_time = None
        self.execution_time = None

    def train(self, train_dataset, patience=5, min_delta=1e-4):
        """
        Trains the LSTM model with early stopping based on training loss.
        
        Args:
            train_dataset: Dataset for training
            patience: Number of epochs to wait before early stopping
            min_delta: Minimum change in loss to be considered as improvement
        """
        self.start_time = time.time()
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float('inf')
        patience_counter = 0
        best_model_path = 'best_model.pth'

        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            self.model.train()
            total_loss = 0
            train_predictions = []
            train_labels = []

            # Training loop
            for sequences, labels in train_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_predictions.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            # Calcular métricas de treino
            avg_loss = total_loss / len(train_loader)
            train_acc = balanced_accuracy_score(train_labels, train_predictions)

            # Atualizar histórico
            self.training_history['train_loss'].append(avg_loss)
            self.training_history['train_accuracy'].append(train_acc)

            # Early stopping check
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
                # Salvar melhor modelo
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            self.logger.info(
                f"Epoch {epoch}: Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}")

            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                # Carregar melhor modelo
                self.model.load_state_dict(torch.load(best_model_path))
                break

            self.training_history['epoch_times'].append(time.time() - epoch_start)

        # Limpar arquivo temporário do modelo
        import os
        if os.path.exists(best_model_path):
            os.remove(best_model_path)

        self.execution_time = time.time() - self.start_time

    def evaluate(self, data_loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in data_loader:
                sequences = sequences.to(self.device)
                outputs = self.model(sequences)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        return balanced_accuracy_score(all_labels, all_preds)

    def predict(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for sequences in test_loader:
                if isinstance(sequences, tuple):
                    sequences = sequences[0]
                sequences = sequences.to(self.device)
                outputs = self.model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, test_dataset):
        """
        Retorna probabilidades para cada classe.
        
        Args:
            test_dataset: Dataset de teste
        Returns:
            np.ndarray: Array com probabilidades para cada classe
        """
        self.model.eval()
        probas = []

        with torch.no_grad():
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

            for sequences in test_loader:
                if isinstance(sequences, tuple):
                    sequences = sequences[0]
                sequences = sequences.to(self.device)

                # Obter logits do modelo
                outputs = self.model(sequences)

                # Aplicar softmax para obter probabilidades
                proba = torch.softmax(outputs, dim=1)
                probas.append(proba.cpu().numpy())

        return np.vstack(probas)
