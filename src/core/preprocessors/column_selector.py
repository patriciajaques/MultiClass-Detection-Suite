import pandas as pd
from typing import List, Dict

class ColumnSelector:
    def __init__(self, data: pd.DataFrame, num_classes: int = 5):
        self.data = data
        self.num_classes = num_classes  # Max number of classes for a column to be considered nominal

    def get_numerical_columns(self) -> List[str]:
        return self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    def get_nominal_columns(self) -> List[str]:
        condition = lambda col: (
            (self.data[col].dtype == 'object' or self.data[col].dtype == 'int64') and
            self.data[col].nunique() < self.num_classes
        )
        return [col for col in self.data.columns if condition(col)]

    def get_ordinal_columns(self) -> List[str]:
        # Assumes ordinal columns have "ordinal" in their name
        return [col for col in self.data.columns if 'ordinal' in col]

    def get_ordinal_categories(self) -> Dict[str, List]:
        ordinal_columns = self.get_ordinal_columns()
        return {col: self.data[col].unique().tolist() for col in ordinal_columns}

    def get_columns_by_regex(self, regex_pattern: str) -> List[str]:
        return self.data.filter(regex=regex_pattern).columns.tolist()
