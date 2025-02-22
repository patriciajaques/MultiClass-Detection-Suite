"""
Copyright (c) 2025 Patricia Jaques
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License. To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
"""

import logging
from typing import Optional, List
import numpy as np
import pandas as pd
from dython.nominal import correlation_ratio


class DataCleaner:
    """
    Responsible for data cleaning operations with configurable strategies.
    Supports both configuration-based and explicit column removal.
    """

    def __init__(self):
        self.logger = logging.getLogger()

    def remove_instances_with_value(self, data: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
        """
        Removes rows where the specified column contains the given value.

        Args:
            data: Input DataFrame
            column: Column name to check
            value: Value to filter out

        Returns:
            DataFrame with matching rows removed
        """
        if column not in data.columns:
            self.logger.warning(f"Column {column} not found in DataFrame")
            return data

        return data[data[column] != value]
    
    def get_columns_to_remove(self, config_manager) -> List[str]:
        """
        Gets list of columns to remove from configuration.
        
        Args:
            config_manager: Configuration manager instance
        
        Returns:
            List of column names to remove
            
        Raises:
            ValueError: If configuration is invalid or missing
        """
        if not config_manager:
            raise ValueError("ConfigManager not provided")

        columns_config = config_manager.get_config('columns_to_remove')
        if not columns_config:
            raise ValueError("'columns_to_remove' configuration not found")

        return self._extract_columns_from_config(columns_config)
    
    def remove_columns(self, data: pd.DataFrame, columns_to_remove: Optional[List[str]], columns_to_keep: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Removes specified columns from DataFrame using explicit list, excluding columns that must be kept.

        Args:
            data: Input DataFrame
            columns_to_remove: List of columns to remove
            columns_to_keep: List of columns that should not be removed

        Returns:
            DataFrame with specified columns removed
            
        Raises:
            ValueError: If columns_to_remove is not provided
        """
        if columns_to_remove is None:
            raise ValueError("Must provide columns_to_remove list")

        if columns_to_keep is None:
            columns_to_keep = []

        # Filter out columns that must be kept
        columns_to_remove = [col for col in columns_to_remove if col not in columns_to_keep]

        existing_columns = self._get_existing_columns(data, columns_to_remove)

        self._log_missing_columns(columns_to_remove, existing_columns)
        return data.drop(columns=existing_columns)
    

    def get_highly_correlated_features(self, df: pd.DataFrame,
                                    target_column: str,
                                    threshold: float = 0.90) -> List[str]:
        """
        Returns a list of features to remove based on correlation analysis,
        keeping one feature from each highly correlated pair and excluding the target variable.
        
        Args:
            df: Input DataFrame
            target_column: Name of the dependent variable to exclude
            threshold: Correlation threshold (default: 0.90)
        
        Returns:
            List of column names recommended for removal
        
        Example:
            >>> df = pd.DataFrame({'target': [1,2,3], 'f1': [1,2,3], 'f2': [1,2,3], 'f3': [4,5,6]})
            >>> get_highly_correlated_features(df, 'target', 0.90)
            ['f2']  # f1 and f2 are correlated, f2 is selected for removal
        """
        # Remove target column and get numeric columns only
        features_df = df.drop(columns=[target_column]
                            ).select_dtypes(include=[np.number])

        # Calculate correlation matrix
        corr_matrix = features_df.corr()

        # Get upper triangle of correlations
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Initialize set of features to remove
        to_drop = set()

        # For each pair of features with correlation above threshold
        for i in range(len(upper_triangle.columns)):
            for j in range(i + 1, len(upper_triangle.columns)):
                if abs(upper_triangle.iloc[i, j]) > threshold:
                    col_i = upper_triangle.columns[i]
                    col_j = upper_triangle.columns[j]

                    # Decide which feature to remove based on correlation with target
                    # Here we keep the feature that has higher correlation with target
                    feature_to_remove = self.select_feature_by_target_correlation(
                        df, col_i, col_j, target_column)

                    to_drop.add(feature_to_remove)

        return list(to_drop)
    
    def select_feature_by_target_correlation(self, df, feature1, feature2, target):
        """
        Selects which feature to keep based on correlation with target variable.
        
        Example:
        If feature1 has 0.7 correlation with target
        and feature2 has 0.4 correlation with target,
        we would keep feature1.
        """
        corr1 = correlation_ratio(df[target], df[feature1])
        corr2 = correlation_ratio(df[target], df[feature2])

        return feature1 if corr1 > corr2 else feature2


    def clean_data(self,
                data: pd.DataFrame,
                target_column: Optional[str] = None,
                undefined_value: str = '?',
                columns_to_remove: Optional[List[str]] = None,
                columns_to_keep: Optional[List[str]] = None,
                handle_multicollinearity: bool = False) -> pd.DataFrame:
        """
        Applies complete cleaning pipeline to the data.

        Args:
            data: DataFrame to clean
            target_column: Column to check for undefined values
            undefined_value: Value that represents undefined
            columns_to_remove: Optional list of columns to remove
            columns_to_keep: Optional list of columns that should not be removed
            handle_multicollinearity: Whether to handle multicollinearity

        Returns:
            Cleaned DataFrame

        Example:
            >>> cleaner = DataCleaner()
            >>> df = pd.DataFrame({'A': [1, '?', 3], 'B': [4, 5, 6]})
            >>> cleaned = cleaner.clean_data(df, target_column='A', columns_to_remove=['B'])
        """
        if data.empty:
            self.logger.warning("Empty DataFrame provided")
            return data.copy()

        cleaned_data = data.copy()

        # Handle undefined values if specified
        if target_column is not None:
            if target_column not in cleaned_data.columns:
                self.logger.warning(
                    f"Undefined value column '{target_column}' not found in DataFrame")
            else:
                cleaned_data = self.remove_instances_with_value(
                    cleaned_data, target_column, undefined_value
                )

        # Remove specified columns if any
        if columns_to_remove:
            cleaned_data = self.remove_columns(
                cleaned_data,
                columns_to_remove=columns_to_remove,
                columns_to_keep=columns_to_keep
            )

        if handle_multicollinearity:
            highly_correlated_features = self.get_highly_correlated_features(
                cleaned_data, target_column=target_column
            )
            cleaned_data = self.remove_columns(
                cleaned_data, columns_to_remove=highly_correlated_features, columns_to_keep=columns_to_keep
            )
        
        return cleaned_data

    def _extract_columns_from_config(self, config: dict) -> List[str]:
        """Extracts column names from configuration dictionary."""
        all_columns = []

        for content in config.values():
            if isinstance(content, list):
                all_columns.extend(content)
            elif isinstance(content, dict):
                all_columns.extend(
                    col for col in content if isinstance(content[col], list)
                )

        if not all_columns:
            raise ValueError("No columns found for removal in configuration")

        return all_columns

    def _get_existing_columns(self, data: pd.DataFrame, columns: List[str]) -> List[str]:
        """Filters list to only include columns that exist in DataFrame."""
        return [col for col in columns if col in data.columns]

    def _log_missing_columns(self, requested: List[str], existing: List[str]) -> None:
        """Logs information about columns that weren't found in DataFrame."""
        if len(existing) < len(requested):
            missing = set(requested) - set(existing)
            self.logger.warning(f"Columns not found in DataFrame: {missing}")
