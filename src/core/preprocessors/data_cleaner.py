class DataCleaner():
    
    @staticmethod
    def remove_instances_with_value(data, column: str, value: str):
        """
        Remove instances where the specified column has the specified value.

        Args:
            data (pd.DataFrame): The input data.
            column (str): The column to check.
            value (str): The value to remove.

        Returns:
            pd.DataFrame: The cleaned data.
        """
        return data[data[column] != value]
    
    @staticmethod
    def remove_columns(data, columns: list):
        """
        Remove the specified columns from the data.

        Args:
            data (pd.DataFrame): The input data.
            columns (list): The columns to remove.

        Returns:
            pd.DataFrame: The cleaned data.
        """
        return data.drop(columns=columns)
    