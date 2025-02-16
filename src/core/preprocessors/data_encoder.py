import logging
import warnings
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    OneHotEncoder,
    MinMaxScaler,
    OrdinalEncoder,
)
from sklearn.compose import ColumnTransformer
from core.preprocessors.column_selector import ColumnSelector


class DataEncoder:
    """
    DataEncoder is designed to facilitate encoding for classification problems by handling both features and target variables.

    Features:
        - Automatically detects and transforms numerical, nominal, and ordinal columns using a custom ColumnSelector.
        - Supports various scaling strategies for numerical columns ('standard', 'minmax', or 'both').
        - Provides separate methods for encoding features when the input DataFrame contains both features and target
          (i.e. using 'fit', 'transform', 'fit_transform') and when it contains only features
          (i.e. using 'fit_features', 'transform_features', 'fit_transform_features').

    Target:
        - Optionally, a target column can be specified at initialization. When provided, if the input DataFrame
          contains the target column, the encoder will separately fit/transform the features (by dropping the target)
          and the target column.
        - If the DataFrame contains only the target column, then only the target is processed.

   """

    def __init__(
        self,
        categorical_threshold: int = 10,
        scaling_strategy: str = "standard",
        select_numerical: bool = True,
        select_nominal: bool = True,
        select_ordinal: bool = False,
        target_column: str = None,  # If provided, this column will be used for target encoding.
    ):
        self.categorical_threshold = categorical_threshold
        self.scaling_strategy = scaling_strategy
        self.select_numerical = select_numerical
        self.select_nominal = select_nominal
        self.select_ordinal = select_ordinal
        self.target_column = target_column

        # Encoder for the target
        self.target_label_encoder = LabelEncoder()
        self._is_fitted_target = (
            False  # Indicates if the target encoder has been fitted
        )

        # Attributes for features
        self._is_fitted = False
        self.column_selector = None
        self.column_transformer = None
        self.numerical_columns = None
        self.nominal_columns = None
        self.ordinal_columns = None
        self.ordinal_categories = None

        self.logger = logging.getLogger()  # Root logger

    def _prepare_features(self, X: pd.DataFrame, drop_target: bool) -> pd.DataFrame:
        """
        Prepares the feature DataFrame.

        Args:
            X (pd.DataFrame): The input DataFrame.
            drop_target (bool): If True and target_column is set, drop the target column.

        Returns:
            pd.DataFrame: DataFrame containing only features.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if drop_target and self.target_column is not None:
            if self.target_column not in X.columns:
                raise ValueError(
                    f"target_column '{self.target_column}' not found in the DataFrame"
                )
            return X.drop(columns=[self.target_column])
        return X

    def initialize_encoder(self):
        """Initializes the transformers for each type of column."""
        transformers = []

        if self.numerical_columns is not None:
            self.logger.info(
                f"\nConfiguring transformation for {len(self.numerical_columns)} numerical columns"
            )
            if self.scaling_strategy == "standard":
                transformers.append(
                    ("num_standard", StandardScaler(), self.numerical_columns)
                )
            elif self.scaling_strategy == "minmax":
                transformers.append(
                    ("num_minmax", MinMaxScaler(), self.numerical_columns)
                )
            elif self.scaling_strategy == "both":
                transformers.append(
                    ("num_standard", StandardScaler(), self.numerical_columns)
                )
                transformers.append(
                    ("num_minmax", MinMaxScaler(), self.numerical_columns)
                )

        if self.nominal_columns is not None:
            self.logger.info(
                f"\nConfiguring transformation for {len(self.nominal_columns)} nominal columns"
            )
            transformers.append(
                (
                    "nom",
                    OneHotEncoder(
                        sparse_output=False, handle_unknown="ignore", drop="first"
                    ),
                    self.nominal_columns,
                )
            )

        if self.ordinal_columns is not None:
            self.logger.info(
                f"\nConfiguring transformation for {len(self.ordinal_columns)} ordinal columns"
            )
            categories = [self.ordinal_categories[col] for col in self.ordinal_columns]
            transformers.append(
                ("ord", OrdinalEncoder(categories=categories), self.ordinal_columns)
            )

        # Preserve original column names
        self.column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

    def select_columns(self, X: pd.DataFrame):
        """
        Selects numerical, nominal, and ordinal columns based on heuristics defined in ColumnSelector.

        Args:
            X (pd.DataFrame): The input DataFrame containing only features.
        """
        self.column_selector = ColumnSelector(X, self.categorical_threshold)

        self.numerical_columns = (
            self.column_selector.get_numerical_columns()
            if self.select_numerical
            else None
        )
        self.nominal_columns = (
            self.column_selector.get_nominal_columns() if self.select_nominal else None
        )

        if self.select_ordinal:
            self.ordinal_columns = self.column_selector.get_ordinal_columns()
            self.ordinal_categories = self.column_selector.get_ordinal_categories()
        else:
            self.ordinal_columns = None
            self.ordinal_categories = None

    # -------------------------------------------------
    # Modified Methods for Combined or Separate Encoding
    # -------------------------------------------------

    def fit(self, X: pd.DataFrame):
        """
        Fits the encoder on the provided DataFrame.

        If the target column is present:
            - When the DataFrame has more than one column, it fits both the features (by dropping the target)
              and the target separately.
            - When the DataFrame contains only the target column, it fits only the target.
        Otherwise, it fits only the features.

        Returns:
            self
        """
        if self.target_column is not None and self.target_column in X.columns:
            if X.shape[1] == 1:
                # Only target column provided
                self.fit_target(X)
            else:
                # Both features and target provided
                features_df = self._prepare_features(X, drop_target=True)
                self.fit_features(features_df)
                self.fit_target(X)
        else:
            # Only features provided
            self.fit_features(X)
        return self

    def transform(self, X: pd.DataFrame):
        """
        Transforms the provided DataFrame.

        If the target column is present:
            - When the DataFrame has more than one column, it transforms both the features (dropping the target)
              and the target separately, returning a tuple (features_transformed, target_transformed).
            - When the DataFrame contains only the target column, it transforms and returns only the target.
        Otherwise, it transforms and returns only the features.
        """
        if self.target_column is not None and self.target_column in X.columns:
            if X.shape[1] == 1:
                # Only target column provided
                return self.transform_target(X)
            else:
                features_transformed = self.transform_features(
                    X.drop(columns=[self.target_column])
                )
                target_transformed = self.transform_target(X)
                return features_transformed, target_transformed
        else:
            return self.transform_features(X)

    def fit_transform(self, X: pd.DataFrame):
        """
        Fits and transforms the provided DataFrame.

        If the target column is present:
            - When the DataFrame has more than one column, it fits and transforms both the features (dropping the target)
              and the target separately, returning a tuple (features_transformed, target_transformed).
            - When the DataFrame contains only the target column, it fits and transforms only the target.
        Otherwise, it fits and transforms only the features.
        """
        if self.target_column is not None and self.target_column in X.columns:
            if X.shape[1] == 1:
                # Only target column provided
                return self.fit_transform_target(X)
            else:
                features_transformed = self.fit_transform_features(
                    X.drop(columns=[self.target_column])
                )
                target_transformed = self.fit_transform_target(X)
                return features_transformed, target_transformed
        else:
            return self.fit_transform_features(X)

    # ------------------------------------------------------
    # Methods for Features (when provided only features)
    # ------------------------------------------------------

    def fit_features(self, X: pd.DataFrame):
        """
        Fits the feature transformers using only the features.
        Assumes that the input DataFrame contains only feature columns.
        """
        X_features = self._prepare_features(X, drop_target=False)
        self.select_columns(X_features)
        self.initialize_encoder()
        self.column_transformer.fit(X_features)
        self._is_fitted = True
        return self

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the features using the fitted feature transformers.
        Assumes that the input DataFrame contains only feature columns.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Features encoder has not been fitted. Use fit_features or fit_transform_features first."
            )

        X_features = self._prepare_features(X, drop_target=False)
        try:
            X_transformed = self.column_transformer.transform(X_features)
            feature_names = self.column_transformer.get_feature_names_out()
            return pd.DataFrame(
                X_transformed, columns=feature_names, index=X_features.index
            )
        except Exception as e:
            self.logger.error(f"Error during feature transformation: {e}")
            raise

    def fit_transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits and transforms the features.
        Assumes that the input DataFrame contains only feature columns.
        """
        self.fit_features(X)
        return self.transform_features(X)

    # ----------------------------------
    # Methods for Target Encoding
    # These methods operate directly on arrays (y) or DataFrames using target_column.
    # ----------------------------------

    def fit_transform_y(self, y):
        """
        Fits and transforms the target.

        Args:
            y: array-like of shape (n_samples,)
        """
        if y is None:
            raise ValueError("y cannot be None")
        if self._is_fitted_target:
            warnings.warn(
                "LabelEncoder has already been fitted. Using fit_transform again."
            )
        y_encoded = self.target_label_encoder.fit_transform(y)
        self._is_fitted_target = True
        return y_encoded

    def transform_y(self, y):
        """Transforms new target data using the fitted encoder."""
        if y is None:
            raise ValueError("y cannot be None")
        if not self._is_fitted_target:
            raise RuntimeError(
                "Target encoder has not been fitted. Use fit_transform_y first."
            )
        return self.target_label_encoder.transform(y)

    def inverse_transform_y(self, y_encoded):
        """Recovers the original target labels from the encoded data."""
        if not self._is_fitted_target:
            raise RuntimeError(
                "Target encoder has not been fitted. Use fit_transform_y first."
            )
        return self.target_label_encoder.inverse_transform(y_encoded)

    def fit_target(self, X: pd.DataFrame):
        """
        Fits the target encoder using the column defined in target_column.

        Args:
            X (pd.DataFrame): The input DataFrame containing the target column.
        """
        if self.target_column is None:
            raise ValueError("target_column was not defined in the constructor.")
        if self.target_column not in X.columns:
            raise ValueError(
                f"target_column '{self.target_column}' not found in the DataFrame"
            )
        y = X[self.target_column]
        self.target_label_encoder.fit(y)
        self._is_fitted_target = True
        return self

    def transform_target(self, X: pd.DataFrame):
        """
        Transforms the target using the fitted encoder.

        Args:
            X (pd.DataFrame): The input DataFrame containing the target column.
        """
        if self.target_column is None:
            raise ValueError("target_column was not defined in the constructor.")
        if self.target_column not in X.columns:
            raise ValueError(
                f"target_column '{self.target_column}' not found in the DataFrame"
            )
        if not self._is_fitted_target:
            raise RuntimeError(
                "Target encoder has not been fitted. Use fit_target or fit() first."
            )
        y = X[self.target_column]
        return self.target_label_encoder.transform(y)

    def fit_transform_target(self, X: pd.DataFrame):
        """
        Fits and transforms the target by extracting the target column from the DataFrame.

        Args:
            X (pd.DataFrame): The input DataFrame containing the target column.
        """
        self.fit_target(X)
        return self.transform_target(X)

    def inverse_transform_target(self, y_encoded):
        """
        Recovers the original target labels from the encoded data.

        Args:
            y_encoded: Encoded target data.
        """
        if not self._is_fitted_target:
            raise RuntimeError(
                "Target encoder has not been fitted. Use fit_target or fit() first."
            )
        return self.target_label_encoder.inverse_transform(y_encoded)

    # ----------------------------------
    # Auxiliary methods
    # ----------------------------------

    def get_n_features(self):
        """Returns the number of unique classes in the target variable."""
        if hasattr(self, "target_label_encoder"):
            return len(self.target_label_encoder.classes_)
        return 0

    def get_feature_mapping(self):
        """Returns the mapping of categorical features after encoding."""
        if not hasattr(self, "column_transformer") or not self.nominal_columns:
            return {}

        mappings = {}
        for col in self.nominal_columns:
            encoder = self.column_transformer.named_transformers_["nom"]
            categories = encoder.categories_[self.nominal_columns.index(col)]
            mappings[col] = dict(enumerate(categories))
        return mappings

    def get_class_mapping(self):
        """Returns the mapping of target classes after encoding."""
        if not hasattr(self, "target_label_encoder") or not hasattr(
            self.target_label_encoder, "classes_"
        ):
            return {}
        return dict(enumerate(self.target_label_encoder.classes_))
