import os

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
            import traceback
            print(traceback.format_exc())
      

    @staticmethod
    def generate_final_report(all_metrics: list[ClassificationModelMetrics]):
        """Generates a consolidated report of all metrics in a structured format."""

        METRIC_COLUMNS = ['balanced_accuracy','f1-score', 'precision', 'recall', 'kappa']
        metrics_df = pd.DataFrame([
            {
                'Model': m.stage_name,
                'CV Score': m.cv_score,
                **{f'{k}-train': m.train_metrics[k] for k in METRIC_COLUMNS},
                **{f'{k}-test': m.test_metrics[k] for k in METRIC_COLUMNS}
            }
            for m in all_metrics
        ])

        metrics_df.to_csv(
            PathManager.get_path('output') / 'consolidated_metrics.csv',
            sep=';',
            index=False
        )
