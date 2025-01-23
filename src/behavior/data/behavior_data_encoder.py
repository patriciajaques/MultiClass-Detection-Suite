import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from core.preprocessors.data_encoder import DataEncoder


class BehaviorDataEncoder(DataEncoder):

    def __init__(self, num_classes=4, create_sequence_id=True):
        if num_classes != 4:
            raise ValueError(
                "BehaviorDataEncoder suporta apenas 4 classes após unificação")
        super().__init__(
            num_classes=num_classes,
            scaling_strategy='standard',
            select_numerical=True,
            select_nominal=True,
            select_ordinal=False
        )
        self.create_sequence_id = create_sequence_id

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma o DataFrame aplicando encoding nas features e opcionalmente criando IDs de sequência.

        Ordem de operações:
        1. Validação do input
        2. Transformação das features (encoding numérico/nominal)
        3. Adição opcional de sequence_id

        Args:
            X: DataFrame com as features originais

        Returns:
            DataFrame transformado com features codificadas e sequence_id opcional

        Raises:
            ValueError: Se input não for DataFrame
            RuntimeError: Se ocorrer erro na transformação
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input deve ser um pandas DataFrame")

        try:

            # Aplica transformações de encoding definidas no fit
            # (standardization para numéricas, one-hot para nominais)
            X_transformed = super().transform(X)

            return X_transformed

        except Exception as e:
            raise RuntimeError(f"Erro durante transformação: {e}")

    def fit(self, X: pd.DataFrame, y=None):
        print(f"Entrada fit - Shape: {X.shape}")

        try:
            super().fit(X)
            print("Fit realizado com sucesso")
            print(
                f"Colunas numéricas: {len(self.numerical_columns) if self.numerical_columns else 0}")
            print(
                f"Colunas nominais: {len(self.nominal_columns) if self.nominal_columns else 0}")
            return self

        except Exception as e:
            raise RuntimeError(f"Erro durante o fit: {e}")

    def fit_transform_y(self, y):
        y_encoded = self.label_encoder.fit_transform(y)
        self._is_fitted = True
        return y_encoded

    def transform_y(self, y):
        return self.label_encoder.transform(y)

    def inverse_transform_y(self, y_encoded):
        return self.label_encoder.inverse_transform(y_encoded)
