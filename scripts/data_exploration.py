import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import preprocessing as pp

def load_data(file_path = '../data/new_logs_labels.csv'):
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

# Função para verificar se uma coluna possui menos de 5 classes
def has_few_classes(column, num_classes=5):
    return column.nunique() < num_classes

def inspect_df (df):
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

def inspect_num_data (df):
        # Exibir estatísticas descritivas do DataFrame
    print('Estatísticas descritivas do DataFrame:')
    print(df.select_dtypes(include=['float64']).describe())
    print('\n')


def inspect_cat_data (df):
    # Exibir o número de instancias para cada classe
    print('Nro de instancias por classe:')
    
    if isinstance(df, pd.Series):
        if df.nunique() < 10:
            for category in df.unique():
                count = (df == category).sum()
                print(f"Categoria: {category}, Contagem: {count}")
        else:
            print(f"Número de categorias: {df.nunique()}")
    else:
        for i, column in enumerate(df.columns, start=1):
            if df[column].nunique() < 5:
                print(f"({i}) {column}:")
                for category in df[column].unique():
                    count = (df[column] == category).sum()
                    print(f"    Categoria: {category}, Contagem: {count}")
            else:
                print(f"({i}) {column}: Número de categorias: {df[column].nunique()}")
    print('\n')

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

def vis_corr_num(df) :
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

def vis_corr_cat(X, y, output_dir='../output/heatmaps', batch_size=25):
    # Criação do diretório de saída se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X_cat = pp.get_categorical_data(X)
    y_cat = pp.get_categorical_data(y)

    n = len(X_cat.columns)
    m = len(y_cat.columns)

    total_plots = n * m
    batches = (total_plots // batch_size) + (total_plots % batch_size != 0)

    plot_count = 0
    for batch in range(batches):
        fig, axs = plt.subplots(min(batch_size, total_plots), 1, figsize=(5, min(batch_size, total_plots)*5))
        for k in range(batch_size):
            if plot_count >= total_plots:
                break
            i, j = divmod(plot_count, m)
            col_x = X_cat.columns[i]
            col_y = y_cat.columns[j]
            contingency_table = pd.crosstab(index=X_cat[col_x], columns=y_cat[col_y])
            sns.heatmap(contingency_table, annot=True, cmap='coolwarm', fmt=".2f", ax=axs[k % batch_size])
            axs[k % batch_size].set_title(f'{col_x} e {col_y}')
            plot_count += 1
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'heatmap_batch_{batch + 1}.png'))
        plt.close(fig)

# Exemplo de chamada da função
# vis_corr_cat(X, y, output_dir='heatmaps', batch_size=25)

