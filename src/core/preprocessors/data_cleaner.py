from typing import Optional, List
import pandas as pd


class DataCleaner:
    """
    Classe responsável por operações de limpeza de dados.
    Suporta limpeza baseada em configuração ou parâmetros explícitos.
    """

    def __init__(self, config_manager = None):
        """
        Inicializa o DataCleaner.
        
        Args:
            config_manager: Opcional - Instância de ConfigManager para carregar configurações
        """
        self.config_manager = config_manager

    def remove_instances_with_value(self, data: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
        """
        Remove instâncias onde a coluna especificada tem o valor especificado.

        Args:
            data: DataFrame com os dados
            column: Nome da coluna para verificar
            value: Valor a ser removido

        Returns:
            DataFrame sem as instâncias que continham o valor especificado
        """
        return data[data[column] != value]

    def get_columns_to_remove(self) -> List[str]:
        """
        Obtém a lista de colunas para remover da configuração.
        
        Returns:
            Lista de nomes de colunas para remover
            
        Raises:
            ValueError: Se ConfigManager não foi fornecido ou configuração é inválida
        """
        if not self.config_manager:
            raise ValueError("ConfigManager não foi fornecido ao DataCleaner")

        columns_config = self.config_manager.get_config('columns_to_remove')
        if not columns_config:
            raise ValueError("Configuração 'columns_to_remove' não encontrada")

        all_columns = []

        for category, content in columns_config.items():
            # Pula keys de metadados como 'description'
            if isinstance(content, list):
                all_columns.extend(content)
            elif isinstance(content, dict):
                columns = [col for col in content if isinstance(
                    content[col], list)]
                all_columns.extend(columns)

        if not all_columns:
            raise ValueError(
                "Nenhuma coluna encontrada para remoção na configuração")

        return all_columns

    def remove_columns(self,
                       data: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       use_config: bool = True) -> pd.DataFrame:
        """
        Remove colunas do DataFrame.
        Pode usar lista explícita de colunas ou configuração.

        Args:
            data: DataFrame de entrada
            columns: Opcional - Lista explícita de colunas para remover
            use_config: Se True, usa configuração do ConfigManager

        Returns:
            DataFrame sem as colunas removidas
            
        Raises:
            ValueError: Se nem columns nem use_config são fornecidos
        """
        if columns is None and not use_config:
            raise ValueError(
                "Deve fornecer lista de colunas ou usar configuração")

        columns_to_remove = columns if columns is not None else self.get_columns_to_remove()

        # Verifica quais colunas realmente existem no DataFrame
        existing_columns = [
            col for col in columns_to_remove if col in data.columns]

        if len(existing_columns) < len(columns_to_remove):
            missing = set(columns_to_remove) - set(existing_columns)
            print(f"Aviso: Colunas não encontradas no DataFrame: {missing}")

        return data.drop(columns=existing_columns)

    def clean_data(self,
                   data: pd.DataFrame,
                   remove_undefined: bool = True,
                   undefined_column: str = None,
                   undefined_value: str = '?',
                   columns_to_remove: Optional[List[str]] = None,
                   use_config: bool = True) -> pd.DataFrame:
        """
        Aplica pipeline completo de limpeza nos dados.

        Args:
            data: DataFrame para limpar
            remove_undefined: Se deve remover valores indefinidos
            undefined_column: Coluna para verificar valores indefinidos
            undefined_value: Valor que representa indefinido
            columns_to_remove: Lista opcional de colunas para remover
            use_config: Se deve usar configuração para remoção de colunas

        Returns:
            DataFrame limpo
        """
        cleaned_data = data.copy()

        # Remove instâncias indefinidas se solicitado
        if remove_undefined and undefined_column:
            cleaned_data = self.remove_instances_with_value(
                cleaned_data, undefined_column, undefined_value
            )

        # Remove colunas
        cleaned_data = self.remove_columns(
            cleaned_data,
            columns=columns_to_remove,
            use_config=use_config
        )

        return cleaned_data
