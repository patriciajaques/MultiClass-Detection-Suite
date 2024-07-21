import os
from datetime import datetime

class FileManager:
    """
    Classe responsável por operações de arquivos, como salvar textos e CSVs.
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
    def save_text_file_with_timestamp(content, filename, directory=None):
        """
        Salva o conteúdo em um arquivo de texto com timestamp no nome do arquivo.
        """
        filename_with_timestamp = FileManager._generate_filename_with_timestamp(filename)
        return FileManager.save_text_file(content, filename_with_timestamp, directory)

    @staticmethod
    def save_csv_file_with_timestamp(dataframe, filename, directory=None):
        """
        Salva o DataFrame em um arquivo CSV com timestamp no nome do arquivo.
        """
        filename_with_timestamp = FileManager._generate_filename_with_timestamp(filename)
        return FileManager.save_csv_file(dataframe, filename_with_timestamp, directory)


    @staticmethod
    def save_text_file(content, filename, directory=None):
        """
        Salva o conteúdo em um arquivo de texto.
        
        Returns:
            file_path: Caminho completo do arquivo salvo.
        """
        if directory:
            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(directory, filename)
        else:
            file_path = filename

        with open(file_path, 'w') as file:
            file.write(content)
        
        return file_path

    @staticmethod
    def save_csv_file(dataframe, filename, directory=None):
        """
        Salva o DataFrame em um arquivo CSV.
        
        Returns:
            file_path: Caminho completo do arquivo salvo.
        """
        if directory:
            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(directory, filename)
        else:
            file_path = filename
        
        dataframe.to_csv(file_path, index=False)
        return file_path
