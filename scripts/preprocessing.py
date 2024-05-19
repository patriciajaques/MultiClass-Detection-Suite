from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import utils


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

def create_preprocessor(numeric_features, categorical_features):
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

