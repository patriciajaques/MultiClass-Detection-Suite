import os
import joblib
from core.logging.file_utils import FileUtils

class ModelManager:
    @staticmethod
    def save_model(model, filename, directory=None):
        directory = FileUtils._create_directory_if_not_exists(directory)
        file_path = FileUtils._generate_filename_with_timestamp(filename)
        file_path = os.path.join(directory, file_path) if directory else file_path
        joblib.dump(model, file_path)
        return file_path

    @staticmethod
    def load_model(filename, directory=None):
        file_path = os.path.join(directory, filename) if directory else filename
        return joblib.load(file_path)

    @classmethod
    def save_all_models(cls, trained_models, directory, prefix='model'):
        saved_models = []
        for model_name, model_info in trained_models.items():
            filename = f"{prefix}_{model_name}.pkl"
            file_path = cls.save_model(model_info['model'], filename, directory)
            saved_models.append(file_path)
            print(f"Model '{model_name}' saved at: {file_path}")
        return saved_models