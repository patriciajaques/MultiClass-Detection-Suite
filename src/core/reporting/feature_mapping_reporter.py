import pandas as pd
from typing import Any, Dict, List, Optional, Union
from core.preprocessors.data_encoder import DataEncoder
from core.utils.path_manager import PathManager


class FeatureMappingReporter:
    def __init__(self):
        self.output_dir = PathManager.get_path('output')

    def log_feature_mappings(self, encoder: DataEncoder, X: Optional[pd.DataFrame] = None,
                             filename_prefix: str = "feature_mapping") -> None:
        """Gera arquivo CSV com mapeamento das features categóricas e suas contagens."""
        mappings = encoder.get_feature_mapping()
        if not mappings:
            return

        rows = []
        for feature, mapping in mappings.items():
            value_counts = self._get_value_counts(
                X, feature) if X is not None else None

            for code, value in mapping.items():
                row = {'feature': feature, 'codigo': code, 'valor': value}
                self._add_counts_to_row(row, value, value_counts, X)
                rows.append(row)

            self._print_distribution(feature, value_counts, X)

        self._save_mapping_csv(rows, filename_prefix)

    def log_target_mappings(self, encoder: DataEncoder, y: Optional[pd.Series] = None,
                            filename_prefix: str = "target_mapping") -> None:
        """
        Gera arquivo CSV com mapeamento das classes target e suas contagens.
        
        Args:
            encoder: DataEncoder contendo o mapeamento das classes
            y: Array ou Series com os dados originais para contagem
            filename_prefix: Prefixo para o nome do arquivo de saída
        """
        mappings = encoder.get_class_mapping()
        if not mappings:
            return

        # Cria DataFrame base com código e nome da classe
        df = pd.DataFrame([
            {'codigo': code, 'classe': value}
            for code, value in mappings.items()
        ])

        if y is not None:
            # Calcula contagens dos valores numéricos
            value_counts = pd.Series(y).value_counts()

            # Mapeia as contagens usando o código numérico
            df['total_instancias'] = df['codigo'].map(value_counts)

            # Calcula porcentagens
            total = value_counts.sum()
            df['porcentagem'] = (df['total_instancias'] /
                                 total * 100).round(2).astype(str) + '%'

            # Ordena por código
            df = df.sort_values('codigo')

            print("\nDistribuição das classes:")
            for _, row in df.iterrows():
                print(f"Classe {row['classe']} (código {row['codigo']}): "
                      f"{row['total_instancias']} instâncias ({row['porcentagem']})")

        self._save_mapping_csv(df, filename_prefix)

    def log_numeric_feature_mappings(self, X: pd.DataFrame,
                                     filename_prefix: str = "numeric_feature_mapping") -> None:
        """
        Gera arquivo CSV com mapeamento das features numéricas.
        
        Para cada variável numérica, identifica se ela é inteira ou real e 
        exibe a faixa de valores (mínimo a máximo).
        
        Args:
            X: DataFrame com as features
            filename_prefix: Prefixo para o nome do arquivo de saída
        """
        if X is None or X.empty:
            print("DataFrame não fornecido ou vazio.")
            return

        # Seleciona apenas as colunas numéricas
        numeric_cols = X.select_dtypes(include=["number"]).columns
        if len(numeric_cols) == 0:
            print("Nenhuma variável numérica encontrada.")
            return

        rows = []
        for col in numeric_cols:
            series = X[col]
            # Verifica se é do tipo inteiro ou real
            tipo = "int" if pd.api.types.is_integer_dtype(series) else "real"
            # Calcula o mínimo e máximo (faixa)
            min_value = series.min()
            max_value = series.max()
            faixa = f"{min_value} a {max_value}"

            # Cria a linha com as informações da variável numérica
            row = {"feature": col, "codigo": tipo, "valor": faixa}
            rows.append(row)

            print(
                f"Variável numérica '{col}' ({tipo}): de {min_value} a {max_value}")

        self._save_mapping_csv(rows, filename_prefix)

    def _get_value_counts(self, data: pd.DataFrame, column: str) -> Optional[pd.Series]:
        """Retorna contagem de valores para uma coluna se ela existir."""
        return data[column].value_counts() if column in data.columns else None

    def _add_counts_to_row(self, row: Dict, value: Any, value_counts: Optional[pd.Series],
                           data: Optional[pd.DataFrame]) -> None:
        """Adiciona contagens e porcentagens a uma linha do DataFrame."""
        if value_counts is not None and data is not None:
            count = value_counts.get(value, 0)
            row['total_instancias'] = count
            row['porcentagem'] = f"{(count / len(data)) * 100:.2f}%"

    def _print_distribution(self, feature: str, value_counts: Optional[pd.Series],
                            data: Optional[pd.DataFrame]) -> None:
        """Imprime distribuição de valores para uma feature."""
        if value_counts is not None and data is not None:
            print(f"\nDistribuição da feature '{feature}':")
            for value, count in value_counts.items():
                percentage = (count / len(data)) * 100
                print(
                    f"Valor '{value}': {count} instâncias ({percentage:.2f}%)")

    def _print_class_distribution(self, row: pd.Series) -> None:
        """Imprime distribuição de uma classe."""
        print(
            f"Classe {row['classe']} (código {row['codigo']}): {row['total_instancias']} instâncias")

    def _save_mapping_csv(self, data: Union[List[Dict], pd.DataFrame], prefix: str) -> None:
        """Salva o mapeamento em um arquivo CSV."""
        filename = f"{prefix}.csv"
        df = pd.DataFrame(data) if isinstance(data, list) else data
        df.to_csv(self.output_dir / filename, sep=';', index=False)
        print(f"\nMapeamento salvo em: {filename}")
