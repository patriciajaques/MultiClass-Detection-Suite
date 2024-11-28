import pandas as pd
from sklearn.preprocessing import LabelEncoder
from core.preprocessors.column_selector import ColumnSelector
from core.preprocessors.data_encoder import DataEncoder

class BehaviorDataEncoder(DataEncoder):
    def __init__(self, num_classes=5):
        # Chame o construtor da classe pai com os parâmetros corretos
        super().__init__(
            num_classes=num_classes,
            scaling_strategy='standard',  # Padronizar features numéricas
            select_numerical=True,        # Selecionar colunas numéricas
            select_nominal=True,          # Selecionar colunas nominais
            select_ordinal=False          # Não usar colunas ordinais
        )
    
    @staticmethod
    def encode_y(y):
        y_encoded = LabelEncoder().fit_transform(y)
        return y_encoded
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Verificações de debug
        print(f"Entrada transform - Shape: {X.shape}")
        
        # Se o encoder não foi ajustado, fazer o fit
        if not hasattr(self, 'column_transformer'):
            self.fit(X)
            
        # Realizar a transformação
        try:
            X_transformed = super().transform(X)
            print(f"Saída transform - Shape: {X_transformed.shape}")
            
            # Verificar se a transformação foi bem sucedida
            if X_transformed.shape[1] == 0:
                raise ValueError("Transformação resultou em DataFrame vazio")
                
            return X_transformed
            
        except Exception as e:
            print(f"Erro durante a transformação: {e}")
            # Se a transformação falhar, retornar os dados originais
            print("Retornando dados originais após erro...")
            return X

    def fit(self, X: pd.DataFrame, y=None):
        # Verificações de debug
        print(f"Entrada fit - Shape: {X.shape}")
        
        # Realizar o fit
        try:
            super().fit(X)
            print("Fit realizado com sucesso")
            print(f"Colunas numéricas: {len(self.numerical_columns) if self.numerical_columns else 0}")
            print(f"Colunas nominais: {len(self.nominal_columns) if self.nominal_columns else 0}")
            return self
            
        except Exception as e:
            print(f"Erro durante o fit: {e}")
            raise
