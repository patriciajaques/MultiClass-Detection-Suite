from sklearn.impute import KNNImputer, SimpleImputer
import pandas as pd
from typing import Dict, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin


class DataImputer(BaseEstimator, TransformerMixin):
    """
    Classe responsável por estratégias de imputação de valores faltantes.
    Herda de BaseEstimator e TransformerMixin para integração com sklearn.
    """

    def __init__(self,
                 numerical_strategy: str = 'knn',
                 categorical_strategy: str = 'most_frequent',
                 knn_neighbors: int = 5,
                 categorical_fill_value: str = 'missing'):
        """
        Args:
            numerical_strategy: Estratégia para features numéricas ('knn', 'mean', 'median')
            categorical_strategy: Estratégia para features categóricas ('most_frequent', 'constant')
            knn_neighbors: Número de vizinhos para KNNImputer
            categorical_fill_value: Valor para preencher categóricas se strategy='constant'
        """
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.knn_neighbors = knn_neighbors
        self.categorical_fill_value = categorical_fill_value

        # Inicializados no fit
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.numerical_columns = None
        self.categorical_columns = None

    def _identify_columns(self, X: pd.DataFrame) -> None:
        """Identifica colunas numéricas e categóricas com missing values."""
        # Identifica colunas com valores faltantes
        missing_columns = X.columns[X.isnull().any()].tolist()

        # Separa por tipo
        self.numerical_columns = X[missing_columns].select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        self.categorical_columns = X[missing_columns].select_dtypes(
            exclude=['int64', 'float64']).columns.tolist()

        self.logger.info(
            f"Colunas numéricas com missing values: {len(self.numerical_columns)}")
        self.logger.info(
            f"Colunas categóricas com missing values: {len(self.categorical_columns)}")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataImputer':
        """
        Ajusta os imputadores aos dados.

        Args:
            X: DataFrame com os dados
            y: Ignorado, mantido para compatibilidade com sklearn
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X deve ser um pandas DataFrame")

        # Identifica colunas com missing values
        self._identify_columns(X)

        # Configura imputador numérico
        if self.numerical_columns:
            if self.numerical_strategy == 'knn':
                self.numerical_imputer = KNNImputer(
                    n_neighbors=self.knn_neighbors,
                    weights='uniform'
                )
            else:
                self.numerical_imputer = SimpleImputer(
                    strategy=self.numerical_strategy
                )

            # Fit no imputador numérico
            if len(self.numerical_columns) > 0:
                self.numerical_imputer.fit(X[self.numerical_columns])

        # Configura imputador categórico
        if self.categorical_columns:
            if self.categorical_strategy == 'constant':
                strategy, fill_value = 'constant', self.categorical_fill_value
            else:
                strategy, fill_value = self.categorical_strategy, None

            self.categorical_imputer = SimpleImputer(
                strategy=strategy,
                fill_value=fill_value
            )

            # Fit no imputador categórico
            if len(self.categorical_columns) > 0:
                self.categorical_imputer.fit(X[self.categorical_columns])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica a imputação nos dados.

        Args:
            X: DataFrame para transformar

        Returns:
            DataFrame com valores imputados
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X deve ser um pandas DataFrame")

        # Cria cópia para não modificar dados originais
        X_imputed = X.copy()

        # Aplica imputação numérica
        if self.numerical_imputer and self.numerical_columns:
            X_imputed[self.numerical_columns] = self.numerical_imputer.transform(
                X[self.numerical_columns]
            )

        # Aplica imputação categórica
        if self.categorical_imputer and self.categorical_columns:
            X_imputed[self.categorical_columns] = self.categorical_imputer.transform(
                X[self.categorical_columns]
            )

        return X_imputed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Ajusta os imputadores e transforma os dados.

        Args:
            X: DataFrame para ajustar e transformar
            y: Ignorado, mantido para compatibilidade

        Returns:
            DataFrame com valores imputados
        """
        return self.fit(X).transform(X)

    def get_missing_info(self, X: pd.DataFrame) -> Dict[str, dict]:
        """
        Retorna informações sobre valores faltantes no DataFrame.

        Args:
            X: DataFrame para analisar

        Returns:
            Dict com informações sobre missing values por coluna
        """
        missing_info = {}

        for column in X.columns:
            missing_count = X[column].isnull().sum()
            if missing_count > 0:
                missing_info[column] = {
                    'missing_count': missing_count,
                    'missing_percentage': (missing_count / len(X)) * 100,
                    'dtype': str(X[column].dtype)
                }

        return missing_info
