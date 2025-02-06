import numpy as np
import pandas as pd
from typing import Union, Tuple
from imblearn.over_sampling import SMOTE


class DataBalancer:
    def __init__(self, random_state: int = 42, logger=None):
        self.random_state = random_state
        self.logger = logger or print  # Usa print se não for fornecido logger

    def balance_data(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            strategy: Union[str, float] = 'auto'
        ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Applies SMOTE for oversampling, accepting the following 'strategy' forms:

        1. String: 'auto', 'minority', 'majority', 'not minority', 'not majority', or 'all'.
        - 'minority': Only oversample the minority classes.
        - 'majority': Only oversample the majority classes.
        - 'all': Oversample all classes except the absolute majority class.
        - 'auto': (default) Oversample all minority classes so that they match the majority class.
        - 'not minority' and 'not majority': Variants recognized by SMOTE to handle different class sets.
        2. Float: Defines the ratio (minority/majority). For example, 0.75 means that the minority classes
        will be oversampled until they reach 75% of the majority class size.
        
        Any other value will result in an error.
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Classification target.
            strategy (str|float): Oversampling strategy.
        
        Returns:
            (X_res, y_res) (pd.DataFrame, pd.Series): Data after oversampling via SMOTE.
        """

        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # Verifica se strategy é uma das strings válidas OU um float
        valid_strings = {
            'auto',
            'minority',
            'majority',
            'not minority',
            'not majority',
            'all'
        }

        if isinstance(strategy, str):
            if strategy not in valid_strings:
                raise ValueError(
                    f"Strategy '{strategy}' inválida. Escolha entre: {valid_strings} "
                    "ou forneça um float."
                )
            sampling_strategy = strategy
        elif isinstance(strategy, float):
            sampling_strategy = strategy  # SMOTE interpretará esse valor como a razão da minoria
        else:
            raise ValueError(
                "strategy deve ser uma string (auto, minority, majority, etc.) ou float ex.: 0.75"
            )

        # Determina k_neighbors com base no menor tamanho de classe
        class_counts = np.bincount(y_array)
        k_neighbors = min(5, min(class_counts) - 1)

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            k_neighbors=k_neighbors
        )

        X_resampled, y_resampled = smote.fit_resample(X_array, y_array)

        # Reconstrói DataFrame/Series com nomes originais
        if isinstance(X, pd.DataFrame):
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        if isinstance(y, pd.Series):
            y_resampled = pd.Series(y_resampled, name=y.name)


        return X_resampled, y_resampled
