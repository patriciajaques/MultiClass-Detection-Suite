from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, cohen_kappa_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.pipeline import Pipeline

from core.reporting.classification_model_metrics import ClassificationModelMetrics


class ModelEvaluator:
    """Responsável por avaliar um único modelo"""

    @staticmethod
    def evaluate_single_model(
                    pipeline: Pipeline,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_test: pd.DataFrame,
                    y_test: pd.Series,
                    stage_name: str) -> ClassificationModelMetrics:
        """Avalia um único modelo e retorna suas métricas"""
        try:
            print(f"\nAvaliando modelo: {stage_name}")

            # Gerar predições
            train_pred, train_prob = ModelEvaluator._get_predictions(pipeline, X_train)
            test_pred, test_prob = ModelEvaluator._get_predictions(pipeline, X_test)

            # Calcular métricas
            train_metrics = ModelEvaluator._compute_metrics(y_train, train_pred)
            test_metrics = ModelEvaluator._compute_metrics(y_test, test_pred)

            # Obter informações sobre features se disponível
            feature_info = ModelEvaluator._get_feature_info(pipeline, X_train)

            # Criar objeto de métricas com estrutura correta
            metrics = ClassificationModelMetrics(
                stage_name=stage_name,
                train_metrics=train_metrics['avg_metrics'],
                test_metrics=test_metrics['avg_metrics'],
                class_report_train=train_metrics['class_report'],
                class_report_test=test_metrics['class_report'],
                confusion_matrix_train=train_metrics['conf_matrix'],
                confusion_matrix_test=test_metrics['conf_matrix'],
                class_labels=list(np.unique(y_train)),
                label_mapping=[{label: i for i, label in enumerate(np.unique(y_train))}],
                feature_info=feature_info
            )

            return metrics
        except Exception as e:
            print(f"Erro ao avaliar modelo {stage_name}: {str(e)}")
            return None

    @staticmethod
    def _get_predictions(pipeline: Pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Gera predições e probabilidades"""
        return pipeline.predict(X), pipeline.predict_proba(X)

    @staticmethod
    def _compute_metrics(y_true: pd.Series,
                         y_pred: np.ndarray) -> Dict[str, pd.DataFrame]:
        """Calcula todas as métricas de classificação"""
        # Gerar relatório de classificação
        class_report = classification_report(y_true, y_pred, output_dict=True)

        # Calcular matriz de confusão
        conf_matrix = confusion_matrix(y_true, y_pred),

        avg_metrics = ModelEvaluator._compute_avg_metrics(y_true, y_pred)

        return {
            'class_report': class_report,
            'conf_matrix': conf_matrix,
            'avg_metrics': avg_metrics,
        }

    @staticmethod
    def _get_feature_info(pipeline: Pipeline, X: pd.DataFrame) -> Dict[str, Any]:
        """Obtém informações sobre as features após seleção/transformação"""
        try:
            if 'feature_selection' not in pipeline.named_steps:
                return {
                    'type': 'original',
                    'n_features': X.shape[1],
                    'description': f'Using all {X.shape[1]} original features'
                }

            selector = pipeline.named_steps['feature_selection']
            X_transformed = selector.transform(X)

            return {
                'type': 'selector',
                'n_features': X_transformed.shape[1],
                'description': f'Using {X_transformed.shape[1]} selected features'
            }

        except Exception as e:
            return {
                'type': 'error',
                'n_features': X.shape[1],
                'description': f'Error getting feature info: {str(e)}'
            }

    @staticmethod 
    def _compute_avg_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        # Calcular métricas médias
        avg_metrics = {
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1-score': f1_score(y_true, y_pred, average='weighted'),
            'kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        return avg_metrics
