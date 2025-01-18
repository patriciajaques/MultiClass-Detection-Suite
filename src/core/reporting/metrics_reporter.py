from core.evaluation.evaluation import Evaluation  
from core.models.model_manager import ModelManager
from core.reporting.report_formatter import ReportFormatter
from core.logging.file_utils import FileUtils

@staticmethod
def evaluate_models(trained_models, X_train, y_train, X_test, y_test):
    return Evaluation.evaluate_all_models(trained_models, X_train, y_train, X_test, y_test)

@staticmethod
def generate_reports(class_metrics_results, avg_metrics_results, directory="../output/", filename_prefix=""):
    """
    Gera relatórios a partir dos resultados da avaliação dos modelos.
    
    Args:
        class_metrics_results: Resultados das métricas por classe
        avg_metrics_results: Resultados das métricas médias
        directory: Diretório para salvar os relatórios
        filename_prefix: Prefixo para os nomes dos arquivos
    """
    # Verifica se os resultados são vazios ou None
    if class_metrics_results is None or avg_metrics_results is None:
        print("Aviso: Não há resultados para gerar relatórios.")
        return
        
    # Verifica se são dicionários vazios
    if not isinstance(class_metrics_results, dict) or not isinstance(avg_metrics_results, dict):
        print("Aviso: Formato inválido dos resultados.")
        return
        
    if len(class_metrics_results) == 0 or len(avg_metrics_results) == 0:
        print("Aviso: Não há resultados para gerar relatórios.")
        return

    # Gerar relatório textual a partir dos resultados de avaliação
    text_report = ReportFormatter.generate_text_report(class_metrics_results, avg_metrics_results)

    # Imprimir ou salvar o relatório
    FileUtils.save_file_with_timestamp(text_report, filename_prefix+"text_report.txt", directory)

    # Gerar DataFrame detalhado dos relatórios por classe
    try:
        class_report_df = ReportFormatter.generate_class_report_dataframe(class_metrics_results)
        FileUtils.save_file_with_timestamp(class_report_df, filename_prefix+"class_report.csv", directory, is_csv=True)
    except ValueError as e:
        print(f"Aviso: Não foi possível gerar o relatório de classes: {str(e)}")

    # Gerar DataFrame resumido dos relatórios de métricas médias
    try:
        avg_metrics_report_df = ReportFormatter.generate_avg_metrics_report_dataframe(avg_metrics_results)
        FileUtils.save_file_with_timestamp(avg_metrics_report_df, filename_prefix+"avg_metrics_report.csv", directory, is_csv=True)
    except ValueError as e:
        print(f"Aviso: Não foi possível gerar o relatório de métricas médias: {str(e)}")

@staticmethod
def save_models(trained_models, model_dir="../models/", filename_prefix=""):
    # Salvar todos os modelos
    saved_models = ModelManager.save_all_models(trained_models, model_dir, filename_prefix)
    print("Modelos salvos:", saved_models)
    return saved_models