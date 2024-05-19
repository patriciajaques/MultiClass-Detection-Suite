from sklearn.model_selection import train_test_split

def determine_features_to_remove(df):
    """
    Retorna apenas as colunas que são features.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
    
    Returns:
        pd.DataFrame: DataFrame contendo apenas as features.
    """
    
    # Selecionar apenas as colunas cujos nomes iniciam com 'traco_', 'estado_', 'comportamento_' e 'ultimo_'
    removed_features = df.loc[:, df.columns.str.startswith('traco_') | df.columns.str.startswith('estado_') | df.columns.str.startswith('comportamento') | df.columns.str.startswith('ultimo_')]
    removed_features = removed_features.drop('ultimo_passo_correto', axis=1)
    return removed_features

def get_personality_features(df):
    """
    Retorna apenas as colunas que são features de personalidade.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
    
    Returns:
        pd.DataFrame: DataFrame contendo apenas as features de personalidade.
    """
    
    # Selecionar apenas as colunas cujos nomes iniciam com 'traco_'
    personality_features = df.loc[:, df.columns.str.startswith('traco_')]
    # Remover as colunas cujos nomes finalizam com '_cat''
    personality_features = personality_features.loc[:, ~personality_features.columns.str.endswith('_cat')]
    return personality_features

def get_behavior_features(df):
    """
    Retorna apenas as colunas que são features de comportamento.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
    
    Returns:
        pd.DataFrame: DataFrame contendo apenas as features de comportamento.
    """
    
    # Selecionar apenas as colunas cujos nomes iniciam com 'comportamento_'
    #behavior_features = df.loc[:, df.columns.str.startswith('comportamento_') | df.columns.str.startswith('ultimo_comportamento_')]
    behavior_features = df.loc[:, df.columns.str.startswith('comportamento')]
    return behavior_features

def split_features_and_target(df):
    """
    Splits the DataFrame into features (X) and target (y).
    
    Parameters:
    - df: pd.DataFrame - The DataFrame containing features and target.
    
    Returns:
    - X: pd.DataFrame - containing the features.
    - y: pd.DataFrame - containing the target.
    """
    removed_features = determine_features_to_remove(df)
    X = df.drop(columns=removed_features.columns.tolist())
    y = get_behavior_features(removed_features)
    return X, y

def split_student_level(data, test_size=0.2, column_name = 'aluno'):
    """
    Splits the DataFrame into student level.
    
    Parameters:
    - df: pd.DataFrame - The DataFrame containing features and target.
    
    Returns:
    - X: pd.DataFrame - containing the features.
    - y: pd.DataFrame - containing the target.
    """
    # Identificar os IDs únicos dos estudantes
    unique_students = data[column_name].unique()
    num_total_students = len(unique_students)

    # Fazer a divisão dos estudantes em conjuntos de treino e teste
    train_students, test_students = train_test_split(unique_students, test_size=0.2, random_state=42)

    num_test_students = len(test_students)

    # Separar os dados com base nos IDs dos estudantes
    train_data = data[data[column_name].isin(train_students)]
    test_data = data[data[column_name].isin(test_students)]

    # Verificar o tamanho dos conjuntos
    print(f'Número total de alunos: {num_total_students}')
    print(f'Número de alunos no conjunto de teste: {num_test_students}')
    print(f'Tamanho do conjunto de treino: {len(train_data)}')
    print(f'Tamanho do conjunto de teste: {len(test_data)}')

    return train_data, test_data