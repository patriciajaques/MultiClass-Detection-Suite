from datetime import datetime
import os
import joblib

class ModelManager:

    @staticmethod
    def dump_model(model, filename, directory=None):
        """
        Salva o modelo treinado em um arquivo.
        
        Returns:
            file_path: Caminho completo do arquivo salvo.
        """
        if directory:
            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(directory, filename)
        else:
            file_path = filename
        
        joblib.dump(model, file_path)
        return file_path

    @staticmethod
    def load_model(filename, directory=None):
        """
        Carrega um modelo salvo a partir de um arquivo.
        
        Returns:
            model: Modelo carregado.
        """
        if directory:
            file_path = os.path.join(directory, filename)
        else:
            file_path = filename
        
        model = joblib.load(file_path)
        return model

    @staticmethod
    def dump_all_models(trained_models, directory, prefix='model'):
        """
        Salva todos os modelos treinados em arquivos individuais com data e hora.
        
        Returns:
            saved_models: Lista de caminhos completos dos arquivos salvos.
        """
        saved_models = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        for model_name, model_info in trained_models.items():
            model = model_info['model']
            filename = f"{prefix}_{model_name}_{timestamp}.pkl"
            file_path = ModelManager.dump_model(model, filename, directory)
            saved_models.append(file_path)
            print(f"Modelo '{model_name}' salvo em: {file_path}")
        
        return saved_models