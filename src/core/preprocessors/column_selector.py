import pandas as pd
from typing import List, Dict


class ColumnSelector:
    def __init__(self, data: pd.DataFrame, num_classes: int = 5):
        self.data = data
        self.num_classes = num_classes
        print(f"\nAnalise inicial do DataFrame:")
        print(f"Total de colunas: {len(data.columns)}")

    def _is_truly_numeric(self, column):
        """Verifica se uma coluna é verdadeiramente numérica"""
        # Se é numérica mas tem poucos valores únicos, provavelmente é categórica
        if self.data[column].dtype in ['int64', 'float64']:
            unique_count = self.data[column].nunique()
            if unique_count < self.num_classes:
                return False
            return True
        return False

    def get_numerical_columns(self) -> List[str]:
        """Retorna colunas que são verdadeiramente numéricas"""
        numerical_columns = [
            col for col in self.data.columns if self._is_truly_numeric(col)]
        if numerical_columns:
            print(
                f"\nColunas numéricas identificadas: {len(numerical_columns)}")
            print(f"Exemplos: {numerical_columns[:5]}...")
        return numerical_columns if numerical_columns else None

    def get_nominal_columns(self) -> List[str]:
        """Retorna colunas categóricas (incluindo numéricas com poucos valores únicos)"""
        def is_nominal(col):
            if self.data[col].dtype == 'object':
                return True
            return (self.data[col].dtype in ['int64', 'float64'] and
                    self.data[col].nunique() < self.num_classes)

        nominal_columns = [col for col in self.data.columns if is_nominal(col)]
        if nominal_columns:
            print(f"\nColunas nominais identificadas: {len(nominal_columns)}")
            print(f"Exemplos: {nominal_columns[:5]}...")
        return nominal_columns if nominal_columns else None

    def get_ordinal_columns(self) -> List[str]:
        def condition(col): return (
            (isinstance(col, int) or 'ordinal' in str(col)) and
            self.data[col].nunique() <= self.num_classes
        )
        ordinal_columns = [col for col in self.data.columns if condition(col)]
        if ordinal_columns:
            print(f"\nColunas ordinais identificadas: {len(ordinal_columns)}")
            print(f"Exemplos: {ordinal_columns[:5]}...")
        return ordinal_columns if ordinal_columns else None

    def get_ordinal_categories(self) -> Dict[str, List]:
        ordinal_columns = self.get_ordinal_columns()
        if not ordinal_columns:
            return None
        ordinal_categories = {col: self.data[col].unique().tolist() for col in ordinal_columns}
        return ordinal_categories if ordinal_categories else None

    def get_columns_by_regex(self, regex_pattern: str) -> List[str]:
        columns_by_regex = self.data.filter(regex=regex_pattern).columns.tolist()
        return columns_by_regex if columns_by_regex else None
