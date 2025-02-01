from typing import Optional, Tuple, Union
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedShuffleSplit, train_test_split
import pandas as pd
from typing import Tuple


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
    def split_stratified_by_groups(data, group_column, target_column, val_size=None, test_size=0.2, random_state=42):
        """
        Realiza split estratificado por grupos e classes usando StratifiedGroupKFold.
        
        Args:
            data: DataFrame com os dados
            group_column: Nome da coluna que identifica os grupos
            target_column: Nome da coluna target para estratificação
            test_size: Proporção dos dados para teste (ex: 0.2 para 20%)
            val_size: Proporção dos dados para validação. Se None, retorna apenas train/test
            random_state: Semente aleatória para reprodutibilidade
        
        Returns:
            Se val_size is None:
                train_data, test_data
            Se val_size não é None:
                train_data, val_data, test_data
        """
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size deve estar entre 0 e 1")
        if val_size is not None and (val_size <= 0 or val_size >= 1):
            raise ValueError("val_size deve estar entre 0 e 1")

        # Calcula número de splits baseado no test_size
        n_splits = int(1/test_size)

        # Configura o splitter
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state)

        # Gera índices do split
        split_indices = list(
            cv.split(data, y=data[target_column], groups=data[group_column]))

        if val_size is None:
            # Usa apenas o primeiro split para train/test
            train_idx, test_idx = split_indices[0]
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            return train_data, test_data
        else:
            # Pega o primeiro split para separar test
            train_val_idx, test_idx = split_indices[0]

            # Cria subconjunto temporário para train/val
            train_val_data = data.iloc[train_val_idx].copy()

            # Faz segundo split para separar train/val
            cv_val = StratifiedGroupKFold(n_splits=int(
                1/val_size), shuffle=True, random_state=random_state)
            train_idx, val_idx = next(cv_val.split(
                train_val_data,
                y=train_val_data[target_column],
                groups=train_val_data[group_column]
            ))

            # Seleciona os dados finais
            train_data = train_val_data.iloc[train_idx]
            val_data = train_val_data.iloc[val_idx]
            test_data = data.iloc[test_idx]

            return train_data, val_data, test_data

    @staticmethod
    def split_data_stratified(data: pd.DataFrame,
                              target_column: str = 'target',
                              test_size: float = 0.2,
                              val_size: Optional[float] = None,
                              random_state: int = 42) -> Union[Tuple[pd.DataFrame, pd.DataFrame],
                                                               Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Divide os dados mantendo a proporção das classes, retornando 2 ou 3 conjuntos.
        Ideal para problemas de classificação como MNIST, onde não há grupos.

        Args:
            data: DataFrame com os dados
            target_column: Nome da coluna target para estratificação
            test_size: Proporção dos dados para teste
            val_size: Proporção dos dados para validação. Se None, retorna apenas train/test
            random_state: Semente aleatória para reprodutibilidade

        Returns:
            Se val_size is None:
                Tuple[pd.DataFrame, pd.DataFrame]: (dados_treino, dados_teste)
            Se val_size não é None:
                Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (dados_treino, dados_validacao, dados_teste)

        Raises:
            ValueError: Se test_size ou val_size estiverem fora do intervalo (0,1)
        """
        # Validação dos parâmetros
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size deve estar entre 0 e 1")
        if val_size is not None and (val_size <= 0 or val_size >= 1):
            raise ValueError("val_size deve estar entre 0 e 1")

        # Split em 2 conjuntos (train/test)
        if val_size is None:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_state
            )
            train_idx, test_idx = next(
                splitter.split(data, data[target_column]))
            return data.iloc[train_idx], None, data.iloc[test_idx]

        # Split em 3 conjuntos (train/val/test)
        else:
            # Primeiro split para separar o conjunto de teste
            first_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_state
            )
            train_val_idx, test_idx = next(
                first_splitter.split(data, data[target_column]))

            # Dados temporários para train/val
            train_val_data = data.iloc[train_val_idx]

            # Segundo split para separar treino e validação
            # Ajusta val_size para considerar apenas os dados de train_val
            adjusted_val_size = val_size / (1 - test_size)

            second_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=adjusted_val_size,
                random_state=random_state
            )
            train_idx, val_idx = next(second_splitter.split(
                train_val_data,
                train_val_data[target_column]
            ))

            return (train_val_data.iloc[train_idx],
                    train_val_data.iloc[val_idx],
                    data.iloc[test_idx])

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
