import pandas as pd
from imblearn.over_sampling import SMOTE
from typing import Tuple

class DataBalancer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def apply_smote(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Verifique se y é um pandas.Series ou numpy.ndarray
        if isinstance(y, pd.Series):
            y_name = y.name
        else:
            y_name = "target"
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=y_name)