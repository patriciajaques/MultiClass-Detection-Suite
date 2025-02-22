"""
Copyright (c) 2025 Patricia Jaques
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License. To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
"""

import pandas as pd

class DataLoader():
    """
    Class to load data from a CSV file in a dataframe.
    """

    @staticmethod
    def load_data(file_path: str, delimiter: str = ',', encoding: str = 'utf-8') -> None:
        """
        Load data from the CSV file into a DataFrame.
        """
        return pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
