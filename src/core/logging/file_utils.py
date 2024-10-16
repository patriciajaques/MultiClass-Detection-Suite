import os
from datetime import datetime
import pandas as pd

class FileUtils:
    @staticmethod
    def save_file(content, filename, directory=None, is_csv=False):
        directory = FileUtils._create_directory_if_not_exists(directory)
        file_path = os.path.join(directory, filename)
        
        if is_csv:
            content.to_csv(file_path, index=False)
        else:
            with open(file_path, 'w') as file:
                file.write(content)
        
        return file_path

    @staticmethod
    def save_file_with_timestamp(content, filename, directory=None, is_csv=False):
        filename_with_timestamp = FileUtils._generate_filename_with_timestamp(filename)
        return FileUtils.save_file(content, filename_with_timestamp, directory, is_csv)

    @staticmethod
    def _create_directory_if_not_exists(directory):
        if directory:
            os.makedirs(directory, exist_ok=True)
        return directory or ""

    @staticmethod
    def _generate_filename_with_timestamp(filename):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        name, ext = os.path.splitext(filename)
        return f"{name}_{timestamp}{ext}"