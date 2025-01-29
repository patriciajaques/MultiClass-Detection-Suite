from pathlib import Path
import joblib
from datetime import datetime
from sklearn.pipeline import Pipeline

from core.utils.path_manager import PathManager


class ModelPersistence:
    """Gerencia o salvamento e carregamento de pipelines de modelos treinados."""
    

    @staticmethod
    def save_model(pipeline: Pipeline, stage_name: str) -> Path:
        """
        Salva o pipeline do modelo treinado.

        Args:
            pipeline: Pipeline scikit-learn treinado
            stage_name: Nome do modelo e seletor para identificação

        Returns:
            Path: Caminho onde o pipeline foi salvo
        """
        models_dir = PathManager.get_path('models')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{stage_name}_{timestamp}.pkl"
        path =  models_dir / filename

        joblib.dump(pipeline, path)
        return path

    @staticmethod
    def load_model(model_path: Path = None) -> Pipeline:
        """
        Carrega um pipeline salvo.

        Args:
            model_path: Caminho para o arquivo do pipeline

        Returns:
            Pipeline: Pipeline scikit-learn carregado
        """
        if model_path is None:
            model_path = ModelPersistence.get_latest_model()
        return joblib.load(model_path)

    @staticmethod
    def get_latest_model(stage_name: str) -> Path:
        """Retorna o pipeline mais recente para um dado nome."""
        models_dir = PathManager.get_path('models')
        models = list(models_dir.glob(f"{stage_name}*.pkl"))
        return max(models, key=lambda x: x.stat().st_mtime) if models else None
