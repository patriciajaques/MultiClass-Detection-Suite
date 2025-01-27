import os
from datetime import datetime
from pathlib import Path

from core.utils.path_manager import PathManager


class FileUtils:
    @staticmethod
    def save_file(content, filename, directory=None, is_csv=False, csv_params=None):
        """
        Salva um arquivo usando o PathManager para gerenciar caminhos.
        
        Args:
            content: Conteúdo a ser salvo
            filename: Nome do arquivo
            directory: Diretório opcional. Se None, usa o diretório de output padrão
            is_csv: Flag indicando se é um arquivo CSV
            csv_params: Parâmetros para salvamento de CSV
        """
        if directory is None:
            # Usa o diretório de output padrão do projeto
            directory = PathManager.get_path('output')
        else:
            # Se fornecido um caminho relativo, considera relativo ao output
            directory = PathManager.get_path('output') / directory

        directory = FileUtils._create_directory_if_not_exists(directory)
        file_path = directory / filename

        if is_csv:
            default_csv_params = {
                'sep': ';',
                'decimal': ',',
                'index': False,
                'float_format': '%.3f'
            }
            if csv_params:
                default_csv_params.update(csv_params)
            content.to_csv(file_path, **default_csv_params)
        else:
            with open(file_path, 'w') as file:
                file.write(content)

        return str(file_path)

    @staticmethod
    def _create_directory_if_not_exists(directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def save_file_with_timestamp(content, filename, directory=None, is_csv=False, csv_params=None):
        filename_with_timestamp = FileUtils._generate_filename_with_timestamp(filename)
        return FileUtils.save_file(content, filename_with_timestamp, directory, is_csv, csv_params)

    @staticmethod
    def _generate_filename_with_timestamp(filename):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        name, ext = os.path.splitext(filename)
        return f"{name}_{timestamp}{ext}"