import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from core.preprocessors.data_encoder import DataEncoder
from core.preprocessors.sequence_handler import SequenceHandler


class BehaviorDataEncoder(DataEncoder):
    def __init__(self, num_classes=4, create_sequence_id=True):
        super().__init__(
            num_classes=num_classes,
            scaling_strategy='standard',
            select_numerical=True,
            select_nominal=True,
            select_ordinal=False
        )
        self.create_sequence_id = create_sequence_id
        self.sequence_handler = SequenceHandler() if create_sequence_id else None
        self.sequence_columns = ['aluno', 'grupo', 'num_dia', 'num_log']

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
            # Cria cópia para não modificar dados originais
            X_copy = X.copy()

            # Aplica transformações de encoding definidas no fit
            # (standardization para numéricas, one-hot para nominais)
            X_transformed = super().transform(X_copy)
            
            # Opcionalmente adiciona coluna de sequence_id 
            # usando grupo, aluno, num_dia e num_log
            if self.create_sequence_id:
                sequence_ids = self.sequence_handler.transform(X_copy)
                X_transformed['sequence_id'] = sequence_ids
                
            return X_transformed
            
        except Exception as e:
            raise RuntimeError(f"Erro durante transformação: {e}")
        
    def fit(self, X: pd.DataFrame, y=None):
        print(f"Entrada fit - Shape: {X.shape}")

        try:
            if self.create_sequence_id:
                self.sequence_handler.fit(X)

            super().fit(X)
            print("Fit realizado com sucesso")
            print(
                f"Colunas numéricas: {len(self.numerical_columns) if self.numerical_columns else 0}")
            print(
                f"Colunas nominais: {len(self.nominal_columns) if self.nominal_columns else 0}")
            return self

        except Exception as e:
            raise RuntimeError(f"Erro durante o fit: {e}")

    def _create_sequence_ids(self, X: pd.DataFrame) -> np.ndarray:
        # Criar sequence_id original
        sequence_ids = (X['grupo'].astype(str) + '_' +
                        X['aluno'].astype(str) + '_' +
                        X['num_dia'].astype(str) + '_' +
                        X['num_log'].astype(str))

        # Converter para hash numérico mantendo ordem
        from hashlib import sha1

        def hash_to_int(x):
            return int(sha1(str(x).encode()).hexdigest(), 16) % (10**10)

        return np.array([hash_to_int(x) for x in sequence_ids])
    
    def fit_transform_y(self, y):
        """
        Unifica comportamentos 3 e 4 antes de fazer o encoding.
        """
        # Substitui comportamentos on-task-resource e on-task-conversation por on-task-out
        y = y.replace(
            ['on-task-resource', 'on-task-conversation'], 'on-task-out')

        # Faz o encoding após a unificação
        y_encoded = self.label_encoder.fit_transform(y)
        self._is_fitted = True
        return y_encoded

    def transform_y(self, y):
        y = y.replace(
            ['on-task-resource', 'on-task-conversation'], 'on-task-out')
        return self.label_encoder.transform(y)

    def inverse_transform_y(self, y_encoded):
        return self.label_encoder.inverse_transform(y_encoded)
