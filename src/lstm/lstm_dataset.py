from torch.utils.data import Dataset
import torch
import numpy as np


class LSTMDataset(Dataset):
    def __init__(self, sequences, labels=None, sequence_length=10):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return sequence, label
        return sequence

    @staticmethod
    def prepare_sequences(df, sequence_length=10, behavior_labels=None, label_encoder=None):
        sequences = []
        labels = []

        feature_cols = [col for col in df.columns
                        if col not in ['aluno', 'num_dia', 'num_log', 'comportamento']]
        
        for _, group in df.groupby('aluno'):
            print(f"Tamanho do grupo: {len(group)}")
            print(f"Índices do grupo: {group.index}")
            # Apenas os primeiros 2 grupos
            break

        for _, group in df.groupby('aluno'):
            data = group.sort_values(['num_dia', 'num_log'])
            feature_data = data[feature_cols].values

            # Criar padding
            padding = np.zeros((sequence_length - 1, len(feature_cols)))
            padded_data = np.vstack([padding, feature_data])

            for i in range(len(feature_data)):
                # Criar sequência
                start_idx = i
                end_idx = i + sequence_length
                sequence = padded_data[start_idx:end_idx]
                sequences.append(sequence)

                # Adicionar label se necessário
                if behavior_labels is not None and label_encoder is not None:
                    # Pegar o comportamento diretamente do DataFrame original
                    current_behavior = data['comportamento'].iloc[i]
                    # Transformar usando o encoder
                    label = label_encoder.transform([current_behavior])[0]
                    labels.append(label)

        return np.array(sequences), np.array(labels) if behavior_labels is not None else None
