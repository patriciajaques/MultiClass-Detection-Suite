from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

from core.evaluation.model_evaluator import ModelEvaluator
from core.reporting.metrics_reporter import MetricsReporter


class Evaluation:

    @staticmethod
    def evaluate_all_models(trained_models, X_train, y_train, X_test, y_test):
        """
        Avalia todos os modelos treinados usando conjuntos de treino e teste.
        """

        if not trained_models:
            print("Aviso: Nenhum modelo foi treinado com sucesso.")
            return {}

        models_metrics = {}
        for model_name, model_info in trained_models.items():
            print(f"\nAvaliando modelo: {model_name}")

            try:
                pipeline = model_info['model']

                metrics = ModelEvaluator.evaluate_single_model(
                    pipeline, X_train, y_train, X_test, y_test, model_name)
                MetricsReporter.generate_stage_report(metrics)
                models_metrics[model_name] = metrics


            except Exception as e:
                print(f"Erro ao avaliar modelo {model_name}: {str(e)}")
                continue

        if not metrics:
            print("Aviso: Nenhum modelo pôde ser avaliado com sucesso.")
            return {}

        return models_metrics

