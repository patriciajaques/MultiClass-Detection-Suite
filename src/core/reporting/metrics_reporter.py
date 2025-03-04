"""
Copyright (c) 2025 Patricia Jaques
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License. To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
"""

import os
import traceback

import pandas as pd
from core.reporting.report_formatter import ReportFormatter
from core.utils.path_manager import PathManager
from core.reporting.classification_model_metrics import ClassificationModelMetrics


class MetricsReporter:
    """Classe responsável por gerar relatórios a partir dos resultados da avaliação dos modelos."""

    @staticmethod
    def generate_stage_report(model_metrics: ClassificationModelMetrics):
        """
        Gera relatório para um único modelo usando um objeto ClassificationModelMetrics.
        Usa conjunto de valiadação para avaliação.
        
        Args:
            model_metrics: Objeto contendo todas as métricas do modelo
        """
        if not model_metrics:
            print("Aviso: Objeto de métricas vazio ou inválido")
            return

        directory = PathManager.get_path('output')
        stage_name = model_metrics.stage_name

        try:
            # Gerar relatório textual
            text_report = ReportFormatter.generate_text_report(model_metrics)
            with open(os.path.join(directory, f"{stage_name}_text_report.txt"), 'w') as f:
                f.write(text_report)

            # Gerar relatório detalhado de classes
            class_report_df = ReportFormatter.generate_class_report_dataframe(
                model_metrics)
            class_report_df.to_csv(
                os.path.join(directory, f"{stage_name}_class_report.csv"),
                sep=';',
                index=False
            )

            # Gerar relatório de métricas médias
            avg_metrics_df = ReportFormatter.generate_avg_metrics_report_dataframe(
                model_metrics)
            avg_metrics_df.to_csv(
                os.path.join(
                    directory, f"{stage_name}_avg_metrics_report.csv"),
                sep=';',
                index=False
            )

            # Gerar matriz de confusão
            confusion_matrix_report = ReportFormatter.generate_confusion_matrix_report(
                model_metrics)
            with open(os.path.join(directory, f"{stage_name}_confusion_matrix.txt"), 'w') as f:
                f.write(confusion_matrix_report)

        except Exception as e:
            print(f"\nErro ao gerar relatórios para {stage_name}: {str(e)}")
            print(traceback.format_exc())
      
    @staticmethod
    def generate_final_report(all_metrics: list[ClassificationModelMetrics]):
        """Generates a consolidated report of all metrics in a structured format."""

        metrics_df = MetricsReporter.assemble_metrics_summary(all_metrics)

        metrics_df.to_csv(
            os.path.join(PathManager.get_path('output'), 'consolidated_metrics.csv'),
            sep=';',
            index=False
        )

    @staticmethod
    def assemble_metrics_summary(all_metrics):
        METRIC_COLUMNS = ['balanced_accuracy','f1-score', 'precision', 'recall', 'kappa']
        metrics_df = pd.DataFrame([
            {
            'Model': m.stage_name,
            'CV Score': m.cv_score,
            **({f'{k}-train': m.train_metrics[k] for k in METRIC_COLUMNS} if m.train_metrics else {}),
            **({f'{k}-val': m.val_metrics[k] for k in METRIC_COLUMNS} if m.val_metrics else {}),
            **({f'{k}-test': m.test_metrics[k] for k in METRIC_COLUMNS} if m.test_metrics else {}),
            }
            for m in all_metrics
        ])
        metrics_df = ReportFormatter.format_dataframe(metrics_df)
        return metrics_df

    @staticmethod
    def export_trials(study, model_name: str, selector_name: str):
        """
        Exporta os trials do estudo do Optuna para um arquivo CSV.
        """
        try:
            trials_df = study.trials_dataframe()
            output_dir = PathManager.get_path('output')
            filename = f"{model_name}_{selector_name}_optuna_trials.csv"
            filepath = output_dir / filename
            trials_df.to_csv(filepath, sep=';', index=False)
            print(f"Trials exportados para: {filepath}")
        except Exception as e:
            print(f"Erro ao exportar trials: {str(e)}")
            print(traceback.format_exc())

    @staticmethod
    def export_cv_results(search_object, model_name: str, selector_name: str):
        try:
            # Extrai os resultados da busca
            cv_results = search_object.cv_results_
            # Converte para DataFrame
            results_df = pd.DataFrame(cv_results)
            output_dir = PathManager.get_path('output')
            filename = f"{model_name}_{selector_name}_cv_results.csv"
            filepath = output_dir / filename
            results_df.to_csv(filepath, sep=';', index=False)
            print(f"CV results exportados para: {filepath}")
        except Exception as e:
            print(f"Erro ao exportar cv_results: {str(e)}")
            print(traceback.format_exc())


