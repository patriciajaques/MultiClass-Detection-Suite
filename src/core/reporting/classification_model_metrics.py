from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import pandas as pd

@dataclass
class ClassificationModelMetrics():
    """Classe de dados para armazenar métricas de um modelo de classificação"""
    stage_name: str # Nome do estágio: algoritmo do modelo e seletor de features. Ex: 'RandomForest_none', 'LogisticRegression_RFE' 
    train_metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    cv_score: Optional[float] = None
    feature_info: Dict[str, Any] = None
    class_report_train: Optional[pd.DataFrame] = None
    class_report_test: Optional[pd.DataFrame] = None
    confusion_matrix_train: Optional[pd.DataFrame] = None
    confusion_matrix_test: Optional[pd.DataFrame] = None
    class_labels: Optional[List[str]] = None
    label_mapping: Optional[List[Dict[str, int]]] = None
    training_type: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
