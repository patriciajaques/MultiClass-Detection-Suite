import logging
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from typing import Tuple, Optional, Union, Dict


class DataBalancer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.logger = logging.getLogger()

    def apply_smote(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        strategy: Union[str, float, Dict] = 'auto',
        target_ratio: float = 1.0
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Applies SMOTE balancing with flexible strategies.

        Args:
            X: Features DataFrame
            y: Target Series
            strategy: Strategy for resampling:
                     - 'auto': Balances all classes to match the majority class
                     - 'minority': Balances only minority classes
                     - float: Proportion of the majority class to aim for (e.g., 0.75)
                     - dict: Specific number of samples for each class
            target_ratio: Target ratio for minority classes (default 1.0 means equal to majority)

        Returns:
            Tuple containing balanced (X, y)
        """
        self.logger.info("\nIniciando SMOTE...")
        self.logger.info(f"Distribuição original das classes:")

        # Antes de converter para numpy, mostramos as distribuições usando pandas
        if isinstance(y, pd.Series):
            self.logger.info(y.value_counts())
        else:
            unique, counts = np.unique(y, return_counts=True)
            self.logger.info(pd.Series(counts, index=unique))

        # Convert data to numpy if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # Calculate class distribution
        class_counts = np.bincount(y_array)
        majority_class_count = np.max(class_counts)

        # Determine sampling strategy
        if isinstance(strategy, dict):
            sampling_strategy = strategy
        elif isinstance(strategy, float):
            target_count = int(majority_class_count * strategy)
            sampling_strategy = {
                i: min(target_count, count) if count < majority_class_count else count
                for i, count in enumerate(class_counts)
            }
        elif strategy == 'minority':
            sampling_strategy = {
                i: int(majority_class_count * target_ratio)
                for i, count in enumerate(class_counts)
                if count < majority_class_count
            }
        else:  # 'auto' ou qualquer outra string
            sampling_strategy = 'auto'

        # Configure SMOTE
        k_neighbors = min(5, min(class_counts) - 1)
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            k_neighbors=k_neighbors
        )

        # Apply SMOTE
        X_resampled, y_resampled = smote.fit_resample(X_array, y_array)

        self.logger.info("\nDistribuição após SMOTE:")
        # Converter para Series para mostrar a distribuição
        y_series = pd.Series(y_resampled)
        self.logger.info(y_series.value_counts())
        self.logger.info(
            f"Shape após SMOTE - X: {X_resampled.shape}, y: {y_resampled.shape}")

        # Retornar como DataFrame/Series mantendo os nomes das colunas originais
        if isinstance(X, pd.DataFrame):
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        if isinstance(y, pd.Series):
            y_resampled = pd.Series(y_resampled, name=y.name)

        return X_resampled, y_resampled
