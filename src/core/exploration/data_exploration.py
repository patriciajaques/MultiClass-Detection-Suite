import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Union, List, Dict

class DataExploration:

    @staticmethod
    def has_few_classes(column: pd.Series, num_classes: int = 5) -> bool:
        return column.nunique() < num_classes

    @staticmethod
    def inspect_dataframe(df: Union[pd.DataFrame, pd.Series]) -> None:
        print('Primeiras linhas do DataFrame:')
        print(df.head())
        print('\nInformações sobre o DataFrame:')
        print(df.info())

    @staticmethod
    def inspect_numeric_data(df: pd.DataFrame) -> None:
        print('Estatísticas descritivas do DataFrame:')
        print(df.select_dtypes(include=['float64', 'int64']).describe())

    @staticmethod
    def inspect_categorical_data(df: Union[pd.DataFrame, pd.Series]) -> None:
        print('Número de instâncias por classe:')
        if isinstance(df, pd.Series):
            DataExploration._print_series_categories(df)
        else:
            for i, column in enumerate(df.columns, start=1):
                print(f"({i}) {column}:")
                DataExploration._print_series_categories(df[column])

    @staticmethod
    def _print_series_categories(series: pd.Series) -> None:
        if series.nunique() < 10:
            for category in series.unique():
                count = (series == category).sum()
                print(f"    Categoria: {category}, Contagem: {count}")
        else:
            print(f"    Número de categorias: {series.nunique()}")

    @staticmethod
    def create_metadata_file(df: pd.DataFrame, file_path: str = 'data/metadata.csv') -> None:
        metadata = df.dtypes
        metadata.to_csv(file_path, header=['data_type'], sep=';')
        print(f'Metadados salvos em {file_path}')

    @staticmethod
    def visualize_histogram(df: pd.DataFrame, num_bins: int = 10, x_range: tuple = (0, 0), y_range: tuple = (0, 0)) -> None:
        axes = df.hist(bins=num_bins, figsize=(20, 15))
        for ax in axes.flatten():
            if x_range != (0, 0):
                ax.set_xlim(x_range)
            if y_range != (0, 0):
                ax.set_ylim(y_range)
        plt.show()

    @staticmethod
    def visualize_correlation_numeric(df: pd.DataFrame) -> None:
        corr = df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Matriz de Correlação')
        plt.show()

    @staticmethod
    def visualize_correlation_categorical(X: pd.DataFrame, y: pd.DataFrame, output_dir: str = '../output/heatmaps', batch_size: int = 25) -> None:
        os.makedirs(output_dir, exist_ok=True)
        X_cat = X.select_dtypes(include=['object', 'category'])
        y_cat = y.select_dtypes(include=['object', 'category'])

        total_plots = len(X_cat.columns) * len(y_cat.columns)
        batches = (total_plots // batch_size) + (total_plots % batch_size != 0)

        for batch in range(batches):
            fig, axs = plt.subplots(min(batch_size, total_plots - batch * batch_size), 1, figsize=(5, min(batch_size, total_plots - batch * batch_size) * 5))
            axs = [axs] if total_plots - batch * batch_size == 1 else axs
            for k, (i, j) in enumerate([(i, j) for i in range(len(X_cat.columns)) for j in range(len(y_cat.columns))][batch * batch_size:(batch + 1) * batch_size]):
                col_x, col_y = X_cat.columns[i], y_cat.columns[j]
                contingency_table = pd.crosstab(index=X_cat[col_x], columns=y_cat[col_y])
                sns.heatmap(contingency_table, annot=True, cmap='coolwarm', fmt=".2f", ax=axs[k])
                axs[k].set_title(f'{col_x} e {col_y}')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'heatmap_batch_{batch + 1}.png'))
            plt.close(fig)

    @staticmethod
    def visualize_feature_target_correlation(X: pd.DataFrame, y: pd.Series, top_n: int = 10, figsize: tuple = (12, 8)) -> None:
        """
        Visualiza a correlação entre as features numéricas de X e o target numérico y.

        Args:
            X (pd.DataFrame): DataFrame contendo as features.
            y (pd.Series): Series contendo o target numérico.
            top_n (int): Número de features com maior correlação absoluta a serem exibidas.
            figsize (tuple): Tamanho da figura para o plot.

        Returns:
            None
        """
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("O target (y) deve ser numérico para calcular correlações.")

        # Selecionar apenas colunas numéricas de X
        X_numeric = X.select_dtypes(include=['float64', 'int64'])

        # Calcular correlações
        correlations = X_numeric.apply(lambda x: x.corr(y) if pd.api.types.is_numeric_dtype(x) else 0)

        # Ordenar correlações por valor absoluto e selecionar top_n
        top_correlations = correlations.abs().nlargest(top_n)

        # Criar um DataFrame com as correlações para facilitar o plotting
        corr_df = pd.DataFrame({'feature': top_correlations.index, 'correlation': correlations[top_correlations.index]})
        corr_df = corr_df.sort_values('correlation', key=abs, ascending=True)

        # Plotar
        plt.figure(figsize=figsize)
        ax = sns.barplot(x='correlation', y='feature', data=corr_df, orient='h')
        ax.axvline(x=0, color='black', linewidth=0.5)
        plt.title(f'Top {top_n} Correlações entre Features e Target')
        plt.xlabel('Correlação')
        plt.ylabel('Feature')
        
        # Adicionar valores de correlação nas barras
        for i, v in enumerate(corr_df['correlation']):
            ax.text(v, i, f'{v:.2f}', va='center', fontweight='bold')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def get_feature_target_correlation(X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Calcula a correlação entre as features numéricas de X e o target numérico y.

        Args:
            X (pd.DataFrame): DataFrame contendo as features.
            y (pd.Series): Series contendo o target numérico.

        Returns:
            pd.Series: Series contendo as correlações ordenadas por valor absoluto.
        """
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("O target (y) deve ser numérico para calcular correlações.")

        X_numeric = X.select_dtypes(include=['float64', 'int64'])
        correlations = X_numeric.apply(lambda x: x.corr(y) if pd.api.types.is_numeric_dtype(x) else 0)
        return correlations.sort_values(key=abs, ascending=False)

