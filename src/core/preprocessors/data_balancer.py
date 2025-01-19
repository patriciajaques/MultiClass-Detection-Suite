import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from typing import Tuple


class DataBalancer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def apply_smote(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        print("\nIniciando SMOTE...")
        print(f"Tipo de X: {type(X)}")
        print(f"Tipo de y: {type(y)}")

        if isinstance(X, pd.DataFrame):
            print("Convertendo X de DataFrame para array...")
            X = X.values
        if isinstance(y, pd.Series):
            print("Convertendo y de Series para array...")
            y = y.values

        print(f"Shape de X após conversão: {X.shape}")
        print(f"Shape de y após conversão: {y.shape}")

        # Verificar se há valores nulos
        if np.isnan(X).any():
            print("AVISO: Valores nulos encontrados em X!")
            X = np.nan_to_num(X)

        smote = SMOTE(random_state=self.random_state)
        try:
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print("SMOTE concluído com sucesso!")
            print(
                f"Shape após SMOTE - X: {X_resampled.shape}, y: {y_resampled.shape}")
        except Exception as e:
            print(f"Erro durante SMOTE: {str(e)}")
            print(f"Valores únicos em y: {np.unique(y)}")
            raise

        # Verifique se y é um pandas.Series ou numpy.ndarray
        if isinstance(y, pd.Series):
            y_name = y.name
        else:
            y_name = "target"

        return pd.DataFrame(X_resampled, columns=range(X.shape[1])), pd.Series(y_resampled, name=y_name)
