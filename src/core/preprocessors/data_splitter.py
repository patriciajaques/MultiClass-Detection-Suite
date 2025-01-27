from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedShuffleSplit
import pandas as pd
from typing import Tuple, Optional


class DataSplitter:
    @staticmethod
    def split_by_groups(data: pd.DataFrame,
                        group_column: str,
                        test_size: float = 0.2,
                        random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide os dados mantendo todas as instâncias de um mesmo grupo juntas.
        Útil para dados agrupados como: estudantes, pacientes, empresas, sensores, etc.

        Args:
            data: DataFrame com os dados
            group_column: Nome da coluna que identifica os grupos
            test_size: Proporção dos dados para teste
            random_state: Semente aleatória para reprodutibilidade

        Returns:
            Tuple contendo (dados_treino, dados_teste)
        """
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(
            splitter.split(data, groups=data[group_column]))
        return data.iloc[train_idx], data.iloc[test_idx]

    @staticmethod
    def split_stratified_by_groups(data: pd.DataFrame,
                                   group_column: str,
                                   target_column: str,
                                   test_size: float = 0.2,
                                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide os dados estratificadamente mantendo grupos juntos.
        Mantém a proporção das classes enquanto preserva a integridade dos grupos.

        Args:
            data: DataFrame com os dados
            group_column: Nome da coluna que identifica os grupos
            target_column: Nome da coluna target para estratificação
            test_size: Proporção dos dados para teste
            random_state: Semente aleatória para reprodutibilidade

        Returns:
            Tuple contendo (dados_treino, dados_teste)
        """
        n_splits = int(1/test_size)
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state)
        train_idx, test_idx = next(splitter.split(
            data, y=data[target_column], groups=data[group_column]
        ))
        return data.iloc[train_idx], data.iloc[test_idx]

    @staticmethod
    def split_data_stratified(data: pd.DataFrame,
                              target_column: str = 'target',
                              test_size: float = 0.2,
                              random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide os dados mantendo a proporção das classes.
        Ideal para problemas de classificação como MNIST, onde não há grupos.

        Args:
            data: DataFrame com os dados
            target_column: Nome da coluna target para estratificação
            test_size: Proporção dos dados para teste
            random_state: Semente aleatória para reprodutibilidade

        Returns:
            Tuple contendo (dados_treino, dados_teste)
        """
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(data, data[target_column]))
        return data.iloc[train_idx], data.iloc[test_idx]

    @staticmethod
    def split_into_x_y(data: pd.DataFrame,
                       target_column: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separa features e target.

        Args:
            data: DataFrame com os dados
            target_column: Nome da coluna target

        Returns:
            Tuple contendo (X, y) onde X são as features e y é o target
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return X, y
