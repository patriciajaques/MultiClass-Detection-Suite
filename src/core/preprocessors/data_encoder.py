import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from core.preprocessors.column_selector import ColumnSelector

class DataEncoder():
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.column_selector = None
        self.column_transformer = None
        self.numerical_columns = None
        self.nominal_columns = None
        self.ordinal_columns = None
        self.ordinal_categories = None

    def initialize_encoder(self):
        transformers = []

        if self.numerical_columns:
            transformers.append(('num', MinMaxScaler(), self.numerical_columns))

        if self.nominal_columns:
            transformers.append(('nom', OneHotEncoder(sparse=False, handle_unknown='ignore'), self.nominal_columns))

        if self.ordinal_columns:
            categories = [self.ordinal_categories[col] for col in self.ordinal_columns]
            transformers.append(('ord', OrdinalEncoder(categories=categories), self.ordinal_columns))

        self.column_transformer = ColumnTransformer(transformers=transformers)

    def select_columns(self, X: pd.DataFrame):
        """ Seleciona colunas numéricas, nominais e ordinais
        baseado em algumas heurísticas genéricas definidas em ColumnSelector.
        """

        self.column_selector = ColumnSelector(X, self.num_classes)
        self.numerical_columns = self.column_selector.get_numerical_columns()
        self.nominal_columns = self.column_selector.get_nominal_columns()
        self.ordinal_columns = self.column_selector.get_ordinal_columns()
        self.ordinal_categories = self.column_selector.get_ordinal_categories()

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

