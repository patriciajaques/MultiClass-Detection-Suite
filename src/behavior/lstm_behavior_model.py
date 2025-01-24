from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



class LSTMBehaviorModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Use only the last output for classification
        out = self.fc(lstm_out[:, -1, :])
        return out


class BehaviorSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]