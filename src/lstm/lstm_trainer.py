import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import balanced_accuracy_score
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

    def train(self, train_dataset, val_dataset=None):
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        if val_dataset:
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        best_val_acc = 0
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            for sequences, labels in train_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            if val_dataset:
                val_acc = self.evaluate(val_loader)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), 'best_model.pth')

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
