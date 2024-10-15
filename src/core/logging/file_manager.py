import os
from core.logging.base_manager import BaseManager

class FileManager(BaseManager):
    @classmethod
    def save_file(cls, content, filename, directory=None, is_csv=False):
        directory = cls._create_directory_if_not_exists(directory)
        file_path = os.path.join(directory, filename)
        
        if is_csv:
            content.to_csv(file_path, index=False)
        else:
            with open(file_path, 'w') as file:
                file.write(content)
        
        return file_path

    @classmethod
    def save_file_with_timestamp(cls, content, filename, directory=None, is_csv=False):
        filename_with_timestamp = cls._generate_filename_with_timestamp(filename)
        return cls.save_file(content, filename_with_timestamp, directory, is_csv)

    @classmethod
    def save_text_file(cls, content, filename, directory=None):
        return cls.save_file(content, filename, directory)

    @classmethod
    def save_csv_file(cls, dataframe, filename, directory=None):
        return cls.save_file(dataframe, filename, directory, is_csv=True)

    @classmethod
    def save_text_file_with_timestamp(cls, content, filename, directory=None):
        return cls.save_file_with_timestamp(content, filename, directory)

    @classmethod
    def save_csv_file_with_timestamp(cls, dataframe, filename, directory=None):
        return cls.save_file_with_timestamp(dataframe, filename, directory, is_csv=True)