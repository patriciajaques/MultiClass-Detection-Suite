from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from IPython.core.debugger import set_trace


import utils
import pandas as pd
from data_exploration import concat_features_and_target 

def get_data_by_type(data, data_type='categorical', num_classes=5):
    """
    Seleciona colunas do DataFrame baseado no tipo de dado e número de classes.
    
    Args:
        data (pd.DataFrame): DataFrame de entrada.
        data_type (str): Tipo de dado para seleção ('categorical' ou 'numerical').
        num_classes (int): Número de classes para considerar uma coluna categórica.
    
    Returns:
        pd.DataFrame: DataFrame com as colunas selecionadas.
    """
    if data_type == 'categorical':
        condition = lambda col: (data[col].dtype == 'object' or data[col].dtype == 'int64') and data[col].nunique() < num_classes
    else:
        condition = lambda col: data[col].dtype in ['float64', 'int64'] and data[col].nunique() >= num_classes
    
    selected_columns = [col for col in data.columns if condition(col)]
    selected_data = data[selected_columns].copy()
    if data_type == 'categorical':
        selected_data = selected_data.astype('category')
    
    return selected_data

from sklearn.preprocessing import LabelEncoder

def encode_single_column(data):
    """
    Aplica LabelEncoder a uma coluna ou série pandas.
    """
    le = LabelEncoder()
    return le.fit_transform(data), le

def encode_categorical_columns(X):
    """
    Aplica LabelEncoder às variáveis categóricas de X e reutiliza encode_single_column.
    """
    X_encoded = X.copy()
    label_encoders = {}
    categorical_columns = X_encoded.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_columns:
        X_encoded[col], le = encode_single_column(X_encoded[col])
        label_encoders[col] = le
    
    return X_encoded, label_encoders

def apply_encoders_to_test_data(X_test, label_encoders):
    """
    Aplica os LabelEncoders salvos de X_train em X_test.
    
    Parâmetros:
    - X_test: DataFrame contendo os dados de teste.
    - label_encoders: Dicionário contendo os LabelEncoders para cada coluna categórica.
    
    Retorna:
    - X_test_encoded: DataFrame com as colunas categóricas codificadas usando os LabelEncoders de X_train.
    """
    X_test_encoded = X_test.copy()
    for col, le in label_encoders.items():
        # Aplica o transform apenas nas colunas que existem em X_test e têm um LabelEncoder correspondente
        if col in X_test_encoded.columns:
            X_test_encoded[col] = le.transform(X_test_encoded[col])
    return X_test_encoded

def load_data(file_path='../data/new_logs_labels.csv'):
    """
    Lê, limpa e retorna os dados de um arquivo CSV.
    """
    df = pd.read_csv(file_path, delimiter=';').query("comportamento != '?'")
    X, y = utils.split_features_and_target(df)
    return X, y['comportamento']

def split_train_test_data(X, y, test_size=0.3, random_state=42):
    """
    Divide os dados em conjuntos de treino e teste.
    """
    data = concat_features_and_target(X, y)
    train_data, test_data = utils.split_data_stratified(data, test_size=test_size, target_column='aluno', n_splits=5, random_state=random_state)
    X_train = train_data.drop(columns=['comportamento'])
    y_train = train_data['comportamento']
    X_test = test_data.drop(columns=['comportamento'])
    y_test = test_data['comportamento']
    return X_train, X_test, y_train.values, y_test.values

def create_preprocessor(X_train):
    """
    Cria um pré-processador para colunas numéricas e categóricas.
    """
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('scaler', MinMaxScaler())]), numeric_features),
        ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])
    return preprocessor

def apply_smote(X_train, y_train):
    """
    Aplica SMOTE para realizar oversampling nos dados de treinamento.
    """
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)