from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import utils
import pandas as pd
from data_exploration import concat_features_and_target 



def get_categorical_data(data) :
    
    # Selecionar colunas que são int64 ou object e possuem menos de 5 classes
    selected_columns = [col for col in data.columns if (data[col].dtype == 'int64' or data[col].dtype == 'object') and utils.has_few_classes(data[col])]

    # Criar um novo DataFrame com as colunas selecionadas
    selected_data = data[selected_columns].copy()

    # Converter colunas para o tipo categórico
    for col in selected_data.columns:
        selected_data[col] = selected_data[col].astype('category')

    # Verificar os tipos das variáveis no novo DataFrame
    return selected_data

def get_numerical_data(data) :
    
    # Selecionar colunas que são float64
    selected_columns = [col for col in data.columns if (data[col].dtype == 'float64' or data[col].dtype == 'int64') and has_few_classes(data[col])==False]
    
    # Criar um novo DataFrame com as colunas selecionadas
    selected_data = data[selected_columns].copy()

    # Verificar os tipos das variáveis no novo DataFrame
    return selected_data

# Função para verificar se uma coluna possui menos de 5 classes
def has_few_classes(column, num_classes=5):
    return column.nunique() < num_classes


def encode_labels(y_train):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    return y_train_encoded, label_encoder

def load_data (file_path = '../data/new_logs_labels.csv'):
    """
    Lê um arquivo CSV com delimitador ';' e inspeciona seu conteúdo.

    Args:
        file_path (str): Caminho para o arquivo CSV.
    
    Returns:
        pd.DataFrame: DataFrame contendo os dados lidos.
    """
    df = pd.read_csv(file_path, delimiter=';')
    X, y = utils.split_features_and_target(df)
    y = y['comportamento']
    X.info()
    y.info()

    return X, y

def split_train_test_data (X, y, test_size=0.3, random_state=42):

    """
    Divide os dados em conjuntos de treino e teste.

    Args:
        X (pd.DataFrame): DataFrame contendo as features.
        y (pd.DataFrame): DataFrame contendo o target.
    
    Returns:
        X_train (pd.DataFrame): DataFrame contendo as features de treino.
        X_test (pd.DataFrame): DataFrame contendo as features de teste.
        y_train (pd.DataFrame): DataFrame contendo o target de treino.
        y_test (pd.DataFrame): DataFrame contendo o target de teste.
    """
    
    # Cria um novo dataframe que contém y concatenado com X
    data = concat_features_and_target(X, y)
    train_data, test_data = utils.split_data_stratified(data, test_size=test_size, target_column='aluno', n_splits=5, random_state=random_state)
    # Separar features e rótulos
    X_train = train_data.drop(columns=['comportamento'])
    y_train = train_data['comportamento']
    X_test = test_data.drop(columns=['comportamento'])
    y_test = test_data['comportamento']
    return X_train, X_test, y_train.values, y_test.values

def create_preprocessor(X_train):

    # Identificar colunas numéricas e categóricas
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

    """
    Cria um ColumnTransformer que aplica StandardScaler às colunas numéricas e OneHotEncoder às colunas categóricas.

    Args:
    numeric_features (list): Lista de nomes de colunas numéricas.
    categorical_features (list): Lista de nomes de colunas categóricas.

    Returns:
    ColumnTransformer: Um ColumnTransformer configurado.
    """
    # Criar pré-processadores para colunas numéricas e categóricas
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combinar transformadores usando ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor
