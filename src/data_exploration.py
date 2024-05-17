import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Lê um arquivo CSV com delimitador ';' e inspeciona seu conteúdo.

    Args:
        file_path (str): Caminho para o arquivo CSV.
    
    Returns:
        pd.DataFrame: DataFrame contendo os dados lidos.
    """
    
    # Ler o arquivo CSV com o delimitador ';'
    df = pd.read_csv(file_path, delimiter=';')

    return df

def inspect_data (df):
    """
    Inspeciona o DataFrame.

    Args:
        df (pd.DataFrame): DataFrame a ser inspecionado.
    """
    
    # Exibir as primeiras linhas do DataFrame
    print('Primeiras linhas do DataFrame:')
    print(df.head())
    print('\n')

    # Exibir informações sobre o DataFrame
    print('Informações sobre o DataFrame:')
    print(df.info())
    print('\n')

    # Exibir estatísticas descritivas do DataFrame
    print('Estatísticas descritivas do DataFrame:')
    print(df.describe())
    print('\n')

    # Exibir a contagem de valores únicos por coluna
    print('Contagem de valores únicos por coluna:')
    print(df.nunique())

def create_metadata_file (df, file_path='data/metadata.csv'):

    """
    Cria um arquivo CSV contendo metadados do DataFrame.

    Args:
        df (pd.DataFrame): DataFrame a ser inspecionado.
    """
    
    # Extrair metadados do DataFrame
    metadata = df.dtypes

    # Salvar metadados em um arquivo CSV
    metadata.to_csv(file_path, header=['data_type'], sep=';')

    print('Metadados salvos em data/metadata.csv')

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
    behavior_features = df.loc[:, df.columns.str.startswith('comportamento_') | df.columns.str.startswith('ultimo_comportamento_')]
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

def concat_features_and_target(X, y):
    """
    Concatena as features (X) e o target (y) em um único DataFrame.

    Args:
        X (pd.DataFrame): DataFrame contendo as features.
        y (pd.DataFrame): DataFrame contendo o target.
    
    Returns:
        pd.DataFrame: DataFrame contendo as features e o target.
    """
    
    # Concatenar as features e o target
    df = pd.concat([X, y], axis=1)
    return df

def vis_histogram(df, num_bins = 10, x_min = 0, x_max = 0, y_min = 0, y_max = 0):
    """
    Visualiza o histograma das colunas do DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
    """
    
    # Criar os subplots do histograma
    axes = df.hist(bins=num_bins, figsize=(20, 15))

    for ax in axes.flatten():
        if x_min != x_max:
            ax.set_xlim([x_min, x_max])
        if y_min != y_max:
            ax.set_ylim([y_min, y_max])

    plt.show()

def vis_corr(df) :
    """
    Visualiza a matriz de correlação do DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
    """
    
    # Calcular a matriz de correlação
    corr = df.corr()
    
    # Plotar a matriz de correlação
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlação')
    plt.show()

# Chamar a função para carregar e inspecionar os dados
if __name__ == "__main__":
    # Caminho para o arquivo CSV
    file_path = 'data/new_logs_labels.csv'
    df = load_data(file_path)
    inspect_data(df)
    create_metadata_file(df)
