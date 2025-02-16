import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,  
        )

        self.attention = nn.Sequential(nn.Linear(hidden_size * 2, 1), nn.Tanh())

        # Dropout adicional antes da camada fully connected
        self.fc_dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)  # shape: (batch, seq_len, hidden_size * 2)

        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # shape: (batch, seq_len, 1)
        attention_out = torch.sum(
            lstm_out * attention_weights, dim=1
        )  # shape: (batch, hidden_size * 2)

        # Dropout before final layer
        attention_out = self.fc_dropout(attention_out)

        # Final classification layer
        output = self.fc(attention_out)

        return output

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs.data, 1)
            return predicted
