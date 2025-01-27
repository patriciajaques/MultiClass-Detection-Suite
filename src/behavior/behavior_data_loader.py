# behavior_data_loader.py

from typing import Optional
import pandas as pd
from core.preprocessors.data_loader import DataLoader

class BehaviorDataLoader(DataLoader):

    @staticmethod
    def get_feature_subset(data: pd.DataFrame, regex_pattern: str) -> pd.DataFrame:
        return data.filter(regex=regex_pattern)

    @staticmethod
    def get_behavior_features(data: pd.DataFrame) -> pd.DataFrame:
        return BehaviorDataLoader.get_feature_subset(data, '^comportamento|^ultimo_comportamento')

    @staticmethod
    def get_personality_features(data: pd.DataFrame) -> pd.DataFrame:
        personality_features = BehaviorDataLoader.get_feature_subset(data, '^traco_')
        return personality_features
    
    @staticmethod
    def get_personality_features_names(data: pd.DataFrame) -> list:
        return BehaviorDataLoader.get_personality_features(data).columns.tolist()

    @staticmethod
    def get_target_column(data: pd.DataFrame, target_name: Optional[str] = None) -> pd.Series:
        target = target_name or 'comportamento'
        if target not in data.columns:
            raise ValueError(f"The column '{target}' does not exist in the dataset.")
        return data[target]

    @staticmethod
    def get_data_info(data: pd.DataFrame) -> dict:
        return {
            'num_samples': len(data),
            'num_features': len(data.columns) - 1,
            'num_classes': data['comportamento'].nunique(),
            'class_distribution': data['comportamento'].value_counts().to_dict(),
            'missing_values': data.isnull().sum().to_dict()
        }