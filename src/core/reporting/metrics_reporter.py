import os
from core.models.model_manager import ModelManager
from core.reporting.report_formatter import ReportFormatter
from core.utils.path_manager import PathManager


@staticmethod
def generate_reports(class_metrics_results, avg_metrics_results, filename_prefix="", force_overwrite=False):
    """Gera relatórios a partir dos resultados da avaliação dos modelos."""
    if not class_metrics_results or not avg_metrics_results:
        print("Aviso: Resultados vazios ou inválidos")
        return

    directory = PathManager.get_path('output')
    print(f"\nGerando relatórios no diretório: {directory}")

    # Garante que o diretório existe
    os.makedirs(directory, exist_ok=True)

    # Gera nomes dos arquivos com caminho completo
    text_report_file = os.path.join(
        directory, f"{filename_prefix}text_report.txt")
    class_report_file = os.path.join(
        directory, f"{filename_prefix}class_report.csv")
    avg_metrics_file = os.path.join(
        directory, f"{filename_prefix}avg_metrics_report.csv")

    print(f"Arquivos a serem gerados:")
    print(f"- Relatório texto: {text_report_file}")
    print(f"- Relatório de classes: {class_report_file}")
    print(f"- Relatório de métricas médias: {avg_metrics_file}")

    try:
        # Gerar relatório textual
        text_report = ReportFormatter.generate_text_report(
            class_metrics_results, avg_metrics_results)
        with open(text_report_file, 'w') as f:
            f.write(text_report)
        print(f"\nRelatório texto gerado: {text_report_file}")

        # Gerar DataFrame detalhado dos relatórios por classe
        class_report_df = ReportFormatter.generate_class_report_dataframe(
            class_metrics_results)
        class_report_df.to_csv(class_report_file, sep=';', index=False)
        print(f"Relatório de classes gerado: {class_report_file}")

        # Gerar DataFrame resumido dos relatórios de métricas médias
        avg_metrics_report_df = ReportFormatter.generate_avg_metrics_report_dataframe(
            avg_metrics_results)
        avg_metrics_report_df.to_csv(avg_metrics_file, sep=';', index=False)
        print(f"Relatório de métricas médias gerado: {avg_metrics_file}")

        # Verificar se os arquivos foram criados
        files_created = all(os.path.exists(f) for f in [
                            text_report_file, class_report_file, avg_metrics_file])
        if files_created:
            print("\nTodos os arquivos foram gerados com sucesso!")
            print(f"Tamanhos dos arquivos:")
            print(f"- Texto: {os.path.getsize(text_report_file)} bytes")
            print(f"- Classes: {os.path.getsize(class_report_file)} bytes")
            print(f"- Métricas: {os.path.getsize(avg_metrics_file)} bytes")
        else:
            print("\nAVISO: Nem todos os arquivos foram gerados corretamente!")

    except Exception as e:
        print(f"\nErro ao gerar relatórios: {str(e)}")
        import traceback
        print(traceback.format_exc())

    if filename_prefix != '_Final_':
        try:
            confusion_matrix_file = os.path.join(
                directory, f"{filename_prefix}confusion_matrix.txt")
            confusion_matrix_report = ReportFormatter.generate_confusion_matrix_report(
                class_metrics_results)
            with open(confusion_matrix_file, 'w') as f:
                f.write(confusion_matrix_report)
            print(
                f"Relatório de matriz de confusão gerado: {confusion_matrix_file}")

        except Exception as e:
            print(f"\nErro ao gerar relatório de matriz de confusão: {str(e)}")


@staticmethod
def save_models(trained_models, model_dir="../models/", filename_prefix=""):
    # Salvar todos os modelos
    saved_models = ModelManager.save_all_models(
        trained_models, model_dir, filename_prefix)
    print("Modelos salvos:", saved_models)
    return saved_models
