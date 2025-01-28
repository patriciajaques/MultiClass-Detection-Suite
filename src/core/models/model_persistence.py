# core/models/model_persistence.py
from pathlib import Path
import joblib
from datetime import datetime

from core.utils.path_manager import PathManager


class ModelPersistence:
    """Gerencia o salvamento e carregamento de modelos treinados."""

    def __init__(self):
        """
        Inicializa o ModelPersistence usando o PathManager para gerenciar diretórios.
        A pasta 'models' será criada dentro do diretório base do projeto.
        """
        self.models_dir = PathManager.get_path('models')

    def save_model(self, model_info: dict, model_name: str) -> Path:
        """Salva o modelo treinado com seu pipeline completo."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{model_name}_{timestamp}.pkl"
        path = self.models_dir / filename

        joblib.dump(model_info, path)
        return path

    def load_model(self, model_path: Path) -> dict:
        """Carrega um modelo salvo."""
        return joblib.load(model_path)
    
    def get_latest_model(self, model_name: str) -> Path:
        """Retorna o modelo mais recente para um dado nome."""
        models = list(self.models_dir.glob(f"{model_name}*.pkl"))
        return max(models, key=lambda x: x.stat().st_mtime) if models else None
