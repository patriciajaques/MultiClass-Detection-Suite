import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from typing import Tuple


class DataBalancer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def apply_smote(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        print("\nIniciando SMOTE...")
        print(f"Distribuição original das classes:")
        print(y.value_counts())

        # Convertendo dados para numpy se necessário
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Calculando a estratégia de balanceamento
        class_counts = np.bincount(y)
        majority_class_count = np.max(class_counts)

        # Definindo o objetivo para cada classe minoritária como 75% da classe majoritária
        target_count = int(majority_class_count * 0.75)

        # Criando dicionário de estratégia
        strategy = {
            i: min(target_count, count) if count < majority_class_count else count
            for i, count in enumerate(class_counts)
        }

        print(f"\nEstratégia de balanceamento:")
        print(f"Contagens alvo por classe: {strategy}")

        # Aplicando SMOTE com a estratégia personalizada
        smote = SMOTE(sampling_strategy=strategy,
                      random_state=self.random_state,
                      k_neighbors=min(5, min(class_counts) - 1))

        X_resampled, y_resampled = smote.fit_resample(X, y)

        print("\nDistribuição após SMOTE:")
        print(pd.Series(y_resampled).value_counts())
        print(
            f"Shape após SMOTE - X: {X_resampled.shape}, y: {y_resampled.shape}")

        return pd.DataFrame(X_resampled, columns=range(X.shape[1])), pd.Series(y_resampled, name="target")
