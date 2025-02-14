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
    def prepare_sequences(df, sequence_length=10, feature_cols=None, target_column='target',
                          group_by_col='aluno', sort_by_cols=['num_dia', 'num_log'], data_encoder=None):
        sequences = []
        labels = []

        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_column]

        # Use o DataEncoder para transformar as features
        if data_encoder is not None:
            df_encoded = data_encoder.transform(df[feature_cols])
            feature_data = df_encoded.values
        else:
            feature_data = df[feature_cols].values

        # Garante que o DataEncoder esteja ajustado para o target
        if target_column is not None and data_encoder is not None:
            if not data_encoder._is_fitted:
                data_encoder.fit_transform_y(df[target_column].unique())
            print(f"Classes codificadas: {data_encoder.get_class_mapping()}")

        for _, group in df.groupby(group_by_col):
            data = group.sort_values(sort_by_cols)

            if data_encoder is not None:
                group_encoded = data_encoder.transform(data[feature_cols])
                feature_data = group_encoded.values
            else:
                feature_data = data[feature_cols].values

            # Criar padding
            padding = np.zeros((sequence_length - 1, feature_data.shape[1]))
            padded_data = np.vstack([padding, feature_data])

            for i in range(len(feature_data)):
                # Criar sequência
                start_idx = i
                end_idx = i + sequence_length
                sequence = padded_data[start_idx:end_idx]
                sequences.append(sequence)

                # Adicionar label se necessário
                if target_column is not None and data_encoder is not None:
                    current_label = data[target_column].iloc[i]
                    # Usar o método transform_y do DataEncoder
                    label = data_encoder.transform_y([current_label])[0]
                    labels.append(label)

        sequences = np.array(sequences)
        labels = np.array(labels) if target_column is not None else None

        if len(sequences) == 0 or (target_column is not None and len(labels) == 0):
            print("Erro: Nenhuma sequência ou label foi criada.")
            return None, None

        return sequences, labels
