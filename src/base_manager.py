from datetime import datetime
import os

class BaseManager:
    """
    Classe base responsável por operações comuns de arquivos.
    """

    @staticmethod
    def _generate_filename_with_timestamp(filename="report.txt"):
        """
        Gera um nome de arquivo com timestamp.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        name, ext = os.path.splitext(filename)
        return f"{name}_{timestamp}{ext}"

    @staticmethod
    def _create_directory_if_not_exists(directory):
        """
        Cria o diretório se ele não existir.
        """
        if directory:
            os.makedirs(directory, exist_ok=True)
            return directory
        return ""