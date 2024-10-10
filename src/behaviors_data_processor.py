from base_data_processor import BaseDataProcessor
import utils
import pandas as pd

class BehaviorsDataProcessor(BaseDataProcessor):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)

    def load_data(self, file_path='../data/new_logs_labels.csv'):
        df = pd.read_csv(file_path, delimiter=';').query("comportamento != '?'")
        X, y = utils.split_features_and_target(df)
        return X, y['comportamento']

    def split_train_test_data(self, X, y, test_size=0.3, random_state=42):
        from data_exploration import concat_features_and_target 
        data = concat_features_and_target(X, y)
        train_data, test_data = utils.split_data_stratified(data, test_size=test_size, target_column='aluno', n_splits=5, random_state=random_state)
        X_train = train_data.drop(columns=['comportamento'])
        y_train = train_data['comportamento']
        X_test = test_data.drop(columns=['comportamento'])
        y_test = test_data['comportamento']
        return X_train, X_test, y_train.values, y_test.values