"""
Copyright (c) 2025 Patricia Jaques
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License. To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ClassificationModelMetrics():
    """Classe de dados para armazenar métricas de um modelo de classificação"""
    stage_name: str
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    cv_score: Optional[float] = None
    feature_info: Dict[str, Any] = None
    class_report_train: Optional[Dict[str, Dict[str, float]]] = None
    class_report_val: Optional[Dict[str, Dict[str, float]]] = None
    class_report_test: Optional[Dict[str, Dict[str, float]]] = None
    confusion_matrix_train: Optional[np.ndarray] = None
    confusion_matrix_val: Optional[np.ndarray] = None
    confusion_matrix_test: Optional[np.ndarray] = None
    class_labels: Optional[List[str]] = None
    label_mapping: Optional[List[Dict[str, int]]] = None
    training_type: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converte o objeto para um dicionário serializável."""
        return {
            'stage_name': self.stage_name,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'test_metrics': self.test_metrics,
            'cv_score': self.cv_score,
            'feature_info': self.feature_info,
            'class_report_train': self.class_report_train,
            'class_report_val': self.class_report_val,
            'class_report_test': self.class_report_test,
            'confusion_matrix_train': self.confusion_matrix_train.tolist() if isinstance(self.confusion_matrix_train, np.ndarray) else None,
            'confusion_matrix_val': self.confusion_matrix_val.tolist() if isinstance(self.confusion_matrix_val, np.ndarray) else None,
            'confusion_matrix_test': self.confusion_matrix_test.tolist() if isinstance(self.confusion_matrix_test, np.ndarray) else None,
            'class_labels': self.class_labels,
            'label_mapping': self.label_mapping,
            'training_type': self.training_type,
            'hyperparameters': self.hyperparameters,
            'execution_time': self.execution_time
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClassificationModelMetrics':
        """Cria uma nova instância a partir de um dicionário."""
        # Converter arrays de volta para numpy se existirem
        conf_matrix_train = np.array(data['confusion_matrix_train']) if data.get(
            'confusion_matrix_train') is not None else None
        conf_matrix_val = np.array(data['confusion_matrix_val']) if data.get(
            'confusion_matrix_val') is not None else None
        conf_matrix_test = np.array(data['confusion_matrix_test']) if data.get(
            'confusion_matrix_test') is not None else None

        return cls(
            stage_name=data.get('stage_name'),
            train_metrics=data.get('train_metrics'),
            val_metrics=data.get('val_metrics'),
            test_metrics=data.get('test_metrics'),
            cv_score=data.get('cv_score'),
            feature_info=data.get('feature_info'),
            class_report_train=data.get('class_report_train'),
            class_report_val=data.get('class_report_val'),
            class_report_test=data.get('class_report_test'),
            confusion_matrix_train=conf_matrix_train,
            confusion_matrix_val=conf_matrix_val,
            confusion_matrix_test=conf_matrix_test,
            class_labels=data.get('class_labels'),
            label_mapping=data.get('label_mapping'),
            training_type=data.get('training_type'),
            hyperparameters=data.get('hyperparameters'),
            execution_time=data.get('execution_time')
        )
