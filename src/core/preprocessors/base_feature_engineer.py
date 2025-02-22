"""
Copyright (c) 2025 Patricia Jaques
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License. To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseFeatureEngineer(ABC):
    """Classe base abstrata para feature engineering."""

    def __init__(self):
        self.new_feature_names = []
        # Atributo para acumular as colunas a serem removidas somente ao final
        self.columns_to_drop = set()

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica transformações de feature engineering."""
        pass

    @abstractmethod
    def get_feature_names(self) -> list:
        """Retorna nomes das features geradas."""
        pass
