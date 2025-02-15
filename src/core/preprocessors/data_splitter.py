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
                        val_size: Optional[float] = None,
                        random_state: int = 42) -> Union[Tuple[pd.DataFrame, pd.DataFrame],
                                                       Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Splits data keeping all instances from the same group together.
        
        Args:
            data: DataFrame containing the data
            group_column: Name of the column that identifies the groups
            test_size: Proportion of data for testing (e.g., 0.2 for 20%)
            val_size: Proportion of data for validation. If None, returns only train/test
            random_state: Random seed for reproducibility
    
        Returns:
            If val_size is None:
                Tuple[pd.DataFrame, pd.DataFrame]: (train_data, test_data)
            If val_size is not None:
                Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_data, val_data, test_data)
        """
        # Validate parameters
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if val_size is not None and not 0 < val_size < 1:
            raise ValueError("val_size must be between 0 and 1")
    
        # Initial split into train and test
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(data, groups=data[group_column]))
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
    
        # If validation set is not required, return binary split
        if val_size is None:
            return train_data, test_data
    
        # Adjust validation size relative to test set size
        adjusted_val_size = val_size / test_size
        
        # Split test set into validation and test
        val_splitter = GroupShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=random_state)
        val_idx, final_test_idx = next(val_splitter.split(test_data, groups=test_data[group_column]))
        
        return train_data, test_data.iloc[val_idx], test_data.iloc[final_test_idx]

    @staticmethod
    def split_stratified_by_groups_and_target(data: pd.DataFrame,
                                             group_column: str,
                                             target_column: str,
                                             test_size: float = 0.2,
                                             val_size: Optional[float] = None,
                                             random_state: int = 42) -> Union[Tuple[pd.DataFrame, pd.DataFrame],
                                                                            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Performs a stratified split by groups and target classes using StratifiedGroupKFold.
        This ensures that the proportion of samples for each class is preserved and 
        all samples from the same group are kept together in the same split.
    
        Args:
            data: DataFrame containing the data
            group_column: Name of the column that identifies the groups
            target_column: Name of the target column for stratification
            test_size: Proportion of data for testing (e.g., 0.2 for 20%)
            val_size: Proportion of data for validation. If None, returns only train/test
            random_state: Random seed for reproducibility
    
        Returns:
            If val_size is None:
                Tuple[pd.DataFrame, pd.DataFrame]: (train_data, test_data)
            If val_size is not None:
                Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_data, val_data, test_data)
    
        Raises:
            ValueError: If test_size or val_size are not between 0 and 1
        """
        # Validate parameters
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if val_size is not None and not 0 < val_size < 1:
            raise ValueError("val_size must be between 0 and 1")
    
        # Calculate number of splits based on test_size
        n_splits = int(1/test_size)
    
        # Configure and perform initial split
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_indices = list(cv.split(data, y=data[target_column], groups=data[group_column]))
        train_idx, test_idx = split_indices[0]
    
        # If no validation set is required, return binary split
        if val_size is None:
            return data.iloc[train_idx], data.iloc[test_idx]
    
        # Adjust validation size relative to training set size
        # Since val_size is a proportion of the total dataset, we need to adjust it
        # to be a proportion of the training set
        adjusted_val_size = val_size / (1 - test_size)
    
        # Create temporary subset for train/val split
        train_data = data.iloc[train_idx].copy()
    
        # Configure and perform validation split
        cv_val = StratifiedGroupKFold(
            n_splits=int(1/adjusted_val_size),
            shuffle=True,
            random_state=random_state
        )
        final_train_idx, val_idx = next(cv_val.split(
            train_data,
            y=train_data[target_column],
            groups=train_data[group_column]
        ))
    
        return (train_data.iloc[final_train_idx],
                train_data.iloc[val_idx],
                data.iloc[test_idx])

    @staticmethod
    def split_stratified_by_target(data: pd.DataFrame,
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
    def separate_features_and_target(data: pd.DataFrame,
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
