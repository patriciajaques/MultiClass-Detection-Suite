import os
import joblib
from core.logging.base_manager import BaseManager

class ModelManager(BaseManager):
    @classmethod
    def save_model(cls, model, filename, directory=None):
        directory = cls._create_directory_if_not_exists(directory)
        file_path = os.path.join(directory, filename)
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
            filename = cls._generate_filename_with_timestamp(f"{prefix}_{model_name}.pkl")
            file_path = cls.save_model(model_info['model'], filename, directory)
            saved_models.append(file_path)
            print(f"Model '{model_name}' saved at: {file_path}")
        return saved_models