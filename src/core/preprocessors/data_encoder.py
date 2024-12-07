import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from core.preprocessors.column_selector import ColumnSelector

class DataEncoder():
    def __init__(self, num_classes: int, scaling_strategy: str = 'standard', select_numerical: bool = False, select_nominal: bool = False, select_ordinal: bool = False):
        self.num_classes = num_classes
        self.scaling_strategy = scaling_strategy
        self.column_selector = None
        self.column_transformer = None
        self.numerical_columns = None
        self.nominal_columns = None
        self.ordinal_columns = None
        self.ordinal_categories = None
        self.select_numerical = select_numerical
        self.select_nominal = select_nominal
        self.select_ordinal = select_ordinal

    def initialize_encoder(self):
        transformers = []

        if self.numerical_columns is not None:
            if self.scaling_strategy == 'standard':
                transformers.append(('num_standard', StandardScaler(), self.numerical_columns))
            elif self.scaling_strategy == 'minmax':
                transformers.append(('num_minmax', MinMaxScaler(), self.numerical_columns))
            elif self.scaling_strategy == 'both':
                transformers.append(('num_standard', StandardScaler(), self.numerical_columns))
                transformers.append(('num_minmax', MinMaxScaler(), self.numerical_columns))

        if self.nominal_columns is not None:
            transformers.append(('nom', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first'), self.nominal_columns))

        if self.ordinal_columns is not None:
            categories = [self.ordinal_categories[col] for col in self.ordinal_columns]
            transformers.append(('ord', OrdinalEncoder(categories=categories), self.ordinal_columns))

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

    def fit(self, X: pd.DataFrame, y=None):
        self.select_columns(X)
        self.initialize_encoder()
        self.column_transformer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = self.column_transformer.transform(X)
        feature_names = self.column_transformer.get_feature_names_out()
        if X_transformed.shape[1] != len(feature_names):
            raise ValueError(f"DataEncoder: Shape of transformed data is {X_transformed.shape}, but got {len(feature_names)} feature names.")
        return pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)
    
    @staticmethod
    def encode_y(y):
        y_encoded = LabelEncoder().fit_transform(y)
        return y_encoded