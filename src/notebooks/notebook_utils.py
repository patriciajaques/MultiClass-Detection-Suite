from core.logging.model_manager import ModelManager
from core.evaluation.evaluation import Evaluation  
from core.logging.report_formatter import ReportFormatter
from core.logging.file_utils import FileUtils

@staticmethod
def evaluate_models(trained_models, X_train, y_train, X_test, y_test):
    return Evaluation.evaluate_all_models(trained_models, X_train, y_train, X_test, y_test)

@staticmethod
def generate_reports(class_metrics_results, avg_metrics_results, directory="../output/", filename_prefix=""):

    # Gerar relatório textual a partir dos resultados de avaliação
    text_report = ReportFormatter.generate_text_report(class_metrics_results, avg_metrics_results)

    # Imprimir ou salvar o relatório
    FileUtils.save_file_with_timestamp(text_report, filename_prefix+"text_report.txt", directory)

    # Gerar DataFrame detalhado dos relatórios por classe
    class_report_df = ReportFormatter.generate_class_report_dataframe(class_metrics_results)

    # Gerar DataFrame resumido dos relatórios de métricas médias
    avg_metrics_report_df = ReportFormatter.generate_avg_metrics_report_dataframe(avg_metrics_results)

    # Salvar os DataFrames como arquivos CSV, se necessário
    FileUtils.save_file_with_timestamp(class_report_df, filename_prefix+"class_report.csv", directory, is_csv=True)
    FileUtils.save_file_with_timestamp(avg_metrics_report_df, filename_prefix+"avg_metrics_report.csv", directory, is_csv=True)

@staticmethod
def save_models(trained_models, model_dir="../models/", filename_prefix=""):
    # Salvar todos os modelos
    saved_models = ModelManager.save_all_models(trained_models, model_dir, filename_prefix)
    print("Modelos salvos:", saved_models)
    return saved_models