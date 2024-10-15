from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

class DataSplitter:
    @staticmethod
    def split_by_student_level(data, test_size=0.2, column_name='aluno'):
        unique_students = data[column_name].unique()
        train_students, test_students = train_test_split(unique_students, test_size=test_size, random_state=42)
        train_data = data[data[column_name].isin(train_students)]
        test_data = data[data[column_name].isin(test_students)]
        return train_data, test_data

    @staticmethod
    def split_by_stratified_student_level(data, test_size=0.2, column_name='aluno', target_column='comportamento', n_splits=10):
        unique_students = data[column_name].unique()
        stratified_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
        y = data.groupby(column_name)[target_column].first().loc[unique_students]
        for train_index, test_index in stratified_split.split(unique_students, y):
            train_students = unique_students[train_index]
            test_students = unique_students[test_index]
            break
        train_data = data[data[column_name].isin(train_students)]
        test_data = data[data[column_name].isin(test_students)]
        return train_data, test_data

    @staticmethod
    def split_data_stratified(data, test_size=0.2, target_column='comportamento', n_splits=1, random_state=42):
        stratified_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        for train_index, test_index in stratified_split.split(data, data[target_column]):
            train_data = data.iloc[train_index]
            test_data = data.iloc[test_index]
            break
        return train_data, test_data

    @staticmethod
    def split_into_x_y(data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return X, y