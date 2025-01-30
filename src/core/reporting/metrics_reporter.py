import os
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
    def generate_final_report():
        """Gera relatório consolidado de todas as métricas."""
        metrics_df = MetricsPersistence.get_all_metrics()

        if metrics_df.empty:
            print("Nenhuma métrica encontrada.")
            return

        print("\n=== Relatório Consolidado de Métricas ===")
        print("\nMétricas médias por modelo:")
        print(metrics_df.groupby('stage_name')[
              ['balanced_accuracy', 'precision', 'recall', 'f1-score']].mean())

        # Salvar relatório
        output_dir = PathManager.get_path('output')
        metrics_df.to_csv(output_dir / 'consolidated_metrics.csv', index=False)
        print(
            f"\nRelatório completo salvo em: {output_dir}/consolidated_metrics.csv")
