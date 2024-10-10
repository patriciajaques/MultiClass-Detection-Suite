from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd

class BaseDataProcessor:

    def __init__(self):
        self.data = self.load_data()

    def load_data(self, file_path='../data/new_logs_labels.csv'):
        df = pd.read_csv(file_path, delimiter=';')
        return df
    
    def split_data(self, df, target_column):
        """
        Faz o split em X e y.
        """

        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y

    def get_data_by_type(self, data, data_type='categorical', num_classes=5):
        if data_type == 'categorical':
            condition = lambda col: (self.data[col].dtype == 'object' or self.data[col].dtype == 'int64') and self.data[col].nunique() < num_classes
        else:
            condition = lambda col: self.data[col].dtype in ['float64', 'int64'] and self.data[col].nunique() >= num_classes
        
        selected_columns = [col for col in self.data.columns if condition(col)]
        selected_data = self.data[selected_columns].copy()
        if data_type == 'categorical':
            selected_data = selected_data.astype('category')
        
        return selected_data

    def encode_single_column(self, data):
        le = LabelEncoder()
        return le.fit_transform(data), le

    def encode_categorical_columns(self, num_classes=5):
        categorical_data = self.get_data_by_type(data_type='categorical', num_classes=num_classes)
        X_encoded = self.data.copy()
        label_encoders = {}
        
        for col in categorical_data.columns:
            X_encoded[col], le = self.encode_single_column(X_encoded[col])
            label_encoders[col] = le
        
        return X_encoded, label_encoders

    def apply_encoders_to_test_data(self, X_test, label_encoders):
        X_test_encoded = X_test.copy()
        for col, le in label_encoders.items():
            if col in X_test_encoded.columns:
                X_test_encoded[col] = le.transform(X_test_encoded[col])
        return X_test_encoded
    
    def create_preprocessor(self):
        numeric_features = self.data.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.data.select_dtypes(include=['object', 'category']).columns
        preprocessor = ColumnTransformer(transformers=[
            ('num', Pipeline([('scaler', MinMaxScaler())]), numeric_features),
            ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
        ])
        return preprocessor

    def apply_smote(self, X_train, y_train):
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X_train, y_train)
