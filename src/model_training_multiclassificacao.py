import utils
import data_exploration as de

def load_data (file_path = '../data/new_logs_labels.csv'):
    """
    Lê um arquivo CSV com delimitador ';' e inspeciona seu conteúdo.

    Args:
        file_path (str): Caminho para o arquivo CSV.
    
    Returns:
        pd.DataFrame: DataFrame contendo os dados lidos.
    """
    
    df = de.load_data('../data/new_logs_labels.csv')
    X, y = utils.split_features_and_target(df)
    y = y['comportamento']
    X.info()
    y.info()

    return X, y

def split_train_test_data (X, y):
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
    data = de.concat_features_and_target(X, y)
    train_data, test_data = utils.split_student_level(data, 0.2, column_name='aluno')
    # Separar features e rótulos
    X_train = train_data.drop(columns=['comportamento'])
    y_train = train_data['comportamento']
    X_test = test_data.drop(columns=['comportamento'])
    y_test = test_data['comportamento']
    return X_train, X_test, y_train, y_test