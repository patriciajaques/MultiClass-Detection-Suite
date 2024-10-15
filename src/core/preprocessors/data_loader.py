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
