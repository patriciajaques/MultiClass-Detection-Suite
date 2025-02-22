"""
Copyright (c) 2025 Patricia Jaques
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License. To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
"""

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
    def prepare_sequences(
        df,
        sequence_length=10,
        feature_cols=None,
        target_column="target",
        group_by_col="aluno",
        sort_by_cols=["num_dia", "num_log"],
        data_encoder=None,  # Parameter kept for compatibility but not used.
        balance_data=True,
    ):
        """
        Prepares temporal sequences from a DataFrame for LSTM training.
        
        This implementation assumes that the features and target have already been encoded 
        during preprocessing, thereby removing the redundancy of reapplying transformations here.
        No encoder is applied within this method.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing the temporal data.
            sequence_length (int): Fixed length for each sequence.
            feature_cols (list, optional): List of feature column names. If None, all columns except 
                the target are used.
            target_column (str, optional): Name of the target column.
            group_by_col (str, optional): Column name to group sequences (e.g., student ID).
            sort_by_cols (list, optional): List of columns used to sort the data within each group.
            data_encoder (optional): This parameter is ignored since encoding is assumed to be done prior.
            balance_data (bool, optional): If True, applies class balancing via DataBalancer.
        
        Returns:
            tuple: A pair (sequences, labels) containing the generated sequences and their corresponding labels.
        """
        sequences = []
        labels = []

        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_column]

        # For each group (e.g., student), create sequences without reapplying encoding.
        for _, group in df.groupby(group_by_col):
            data = group.sort_values(sort_by_cols)
            feature_data = data[feature_cols].values  # Uses features already transformed.
            # Create padding with zeros to complete the sequence.
            padding = np.zeros((sequence_length - 1, feature_data.shape[1]))
            padded_data = np.vstack([padding, feature_data])

            for i in range(len(feature_data)):
                start_idx = i
                end_idx = i + sequence_length
                sequence = padded_data[start_idx:end_idx]
                sequences.append(sequence)
                if target_column is not None:
                    # Uses the target value as already encoded.
                    labels.append(data[target_column].iloc[i])

        # Apply class balancing if requested.
        if target_column is not None and balance_data:
            sequences, labels = LSTMDataset.balance_sequences(
                sequences, np.array(labels), strategy="oversample", random_state=42
            )
            if len(sequences) == 0 or len(labels) == 0:
                print("Error: No sequences or labels were generated.")
                return None, None

        return np.array(sequences), np.array(labels)

    @staticmethod
    def balance_sequences(sequences, labels, strategy="oversample", random_state=42):
        labels = labels.astype(np.int64)
        unique_labels, counts = np.unique(labels, return_counts=True)
        target_count = counts.max() if strategy == "oversample" else counts.min()
        balanced_sequences = []
        balanced_labels = []
        np.random.seed(random_state)
        for label in unique_labels:
            idxs = np.where(labels == label)[0]
            current_sequences = np.array(sequences)[idxs]
            current_labels = labels[idxs]
            if strategy == "oversample":
                n_samples = target_count - len(current_labels)
                if n_samples > 0:
                    extra_idx = np.random.choice(
                        len(current_labels), size=n_samples, replace=True
                    )
                    extra_sequences = current_sequences[extra_idx]
                    extra_labels = current_labels[extra_idx]
                    current_sequences = np.concatenate(
                        [current_sequences, extra_sequences], axis=0
                    )
                    current_labels = np.concatenate(
                        [current_labels, extra_labels], axis=0
                    )
            elif strategy == "undersample":
                if len(current_labels) > target_count:
                    choice_idx = np.random.choice(
                        len(current_labels), size=target_count, replace=False
                    )
                    current_sequences = current_sequences[choice_idx]
                    current_labels = current_labels[choice_idx]
            balanced_sequences.append(current_sequences)
            balanced_labels.append(current_labels)
        balanced_sequences = np.concatenate(balanced_sequences, axis=0)
        balanced_labels = np.concatenate(balanced_labels, axis=0)
        perm = np.random.permutation(len(balanced_labels))
        balanced_sequences = balanced_sequences[perm]
        balanced_labels = balanced_labels[perm]
        return balanced_sequences, balanced_labels