import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from core.preprocessors.column_selector import ColumnSelector

class DataEncoder():
    def __init__(self, num_classes: int, scaling_strategy: str = 'standard', select_numerical: bool = True, select_nominal: bool = True, select_ordinal: bool = True):
        self.num_classes = num_classes
        self.scaling_strategy = scaling_strategy
        self.select_numerical = select_numerical
        self.select_nominal = select_nominal
        self.select_ordinal = select_ordinal
        self.label_encoder = LabelEncoder()
        self._is_fitted = False
        self.column_selector = None
        self.column_transformer = None
        self.numerical_columns = None
        self.nominal_columns = None
        self.ordinal_columns = None
        self.ordinal_categories = None


    def initialize_encoder(self):
        """Inicializa os transformadores para cada tipo de coluna."""
        transformers = []

        if self.numerical_columns is not None:
            print(
                f"\nConfigurando transformação para {len(self.numerical_columns)} colunas numéricas")
            if self.scaling_strategy == 'standard':
                transformers.append(
                    ('num_standard', StandardScaler(), self.numerical_columns))
            elif self.scaling_strategy == 'minmax':
                transformers.append(
                    ('num_minmax', MinMaxScaler(), self.numerical_columns))
            elif self.scaling_strategy == 'both':
                transformers.append(
                    ('num_standard', StandardScaler(), self.numerical_columns))
                transformers.append(
                    ('num_minmax', MinMaxScaler(), self.numerical_columns))

        if self.nominal_columns is not None:
            print(
                f"\nConfigurando transformação para {len(self.nominal_columns)} colunas nominais")
            transformers.append(('nom', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first'),
                                self.nominal_columns))

        if self.ordinal_columns is not None:
            print(
                f"\nConfigurando transformação para {len(self.ordinal_columns)} colunas ordinais")
            categories = [self.ordinal_categories[col]
                        for col in self.ordinal_columns]
            transformers.append(('ord', OrdinalEncoder(
                categories=categories), self.ordinal_columns))

        self.column_transformer = ColumnTransformer(transformers=transformers)

    def select_columns(self, X: pd.DataFrame):
        """ Seleciona colunas numéricas, nominais e ordinais
        baseado em algumas heurísticas genéricas definidas em ColumnSelector.
    
        Args:
            X (pd.DataFrame): O DataFrame de entrada.
            select_numerical (bool): Se True, seleciona colunas numéricas.
            select_nominal (bool): Se True, seleciona colunas nominais.
            select_ordinal (bool): Se True, seleciona colunas ordinais.
        """
        self.column_selector = ColumnSelector(X, self.num_classes)
        
        if self.select_numerical:
            self.numerical_columns = self.column_selector.get_numerical_columns()
        else:
            self.numerical_columns = None
    
        if self.select_nominal:
            self.nominal_columns = self.column_selector.get_nominal_columns()
        else:
            self.nominal_columns = None
    
        if self.select_ordinal:
            self.ordinal_columns = self.column_selector.get_ordinal_columns()
            self.ordinal_categories = self.column_selector.get_ordinal_categories()
        else:
            self.ordinal_columns = None
            self.ordinal_categories = None

    def fit(self, X: pd.DataFrame):
        self.select_columns(X)
        self.initialize_encoder()
        self.column_transformer.fit(X)
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        print(f"\nInicio da transformação - Shape entrada: {X.shape}")

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input deve ser um pandas DataFrame")

        try:
            X_transformed = self.column_transformer.transform(X)
            feature_names = self.column_transformer.get_feature_names_out()

            print(f"Fim da transformação - Shape saída: {X_transformed.shape}")
            print(f"Total de features após transformação: {len(feature_names)}")
            print("Distribuição por tipo:")
            print(
                f"- Features numéricas transformadas: {sum(1 for name in feature_names if 'num_' in name)}")
            print(
                f"- Features nominais transformadas: {sum(1 for name in feature_names if 'nom_' in name)}")
            print(
                f"- Features ordinais transformadas: {sum(1 for name in feature_names if 'ord_' in name)}")

            return pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

        except Exception as e:
            print(f"Erro durante transformação: {e}")
            raise

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

    def fit_transform_y(self, y):
        """
        Fit e transforma o target.
        
        Args:
            y: array-like de shape (n_samples,)
            
        Returns:
            array-like: Target codificado
        """
        # Validação
        if y is None:
            raise ValueError("y não pode ser None")

        if self._is_fitted:
            warnings.warn(
                "LabelEncoder já foi ajustado. Usando fit_transform novamente.")

        # Encoding
        y_encoded = self.label_encoder.fit_transform(y)
        self._is_fitted = True

        # Validação pós-encoding
        unique_encoded = np.unique(y_encoded)
        if len(unique_encoded) > self.num_classes:
            raise ValueError(
                f"Número de classes ({len(unique_encoded)}) maior que o esperado ({self.num_classes})"
            )

        return y_encoded

    def transform_y(self, y):
        """Transforma novos dados usando o encoding aprendido"""
        if not self._is_fitted:
            raise RuntimeError(
                "Encoder não foi ajustado. Use fit_transform_y primeiro.")
        return self.label_encoder.transform(y)

    def inverse_transform_y(self, y_encoded):
        """Recupera labels originais"""
        if not self._is_fitted:
            raise RuntimeError(
                "Encoder não foi ajustado. Use fit_transform_y primeiro.")
        return self.label_encoder.inverse_transform(y_encoded)
