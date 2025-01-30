# core/reporting/metrics_persistence.py
import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from core.utils.path_manager import PathManager
from core.reporting.classification_model_metrics import ClassificationModelMetrics

import pandas as pd
from pathlib import Path
from core.utils.path_manager import PathManager
from core.reporting.classification_model_metrics import ClassificationModelMetrics


class MetricsPersistence:
    @staticmethod
    def save_metrics(metrics: ClassificationModelMetrics, stage_name: str) -> Path:
        """
        Salva métricas em formato JSON usando Pandas.
        
        Args:
            metrics: Objeto ClassificationModelMetrics para salvar
            stage_name: Nome do estágio para identificação do arquivo
            
        Returns:
            Path: Caminho onde o arquivo foi salvo
        """
        metrics_dir = PathManager.get_path('metrics')
        metrics_dict = metrics.to_dict()

        filename = f"{stage_name}_metrics.json"
        path = metrics_dir / filename

        # Usar pandas para serializar, que lida automaticamente com tipos numpy
        pd.Series(metrics_dict).to_json(
            path, orient='index', default_handler=str)
        return path

    @staticmethod
    def load_metrics(stage_name: str) -> Optional[ClassificationModelMetrics]:
        """
        Carrega métricas do JSON usando Pandas.
        
        Args:
            stage_name: Nome do estágio para identificar o arquivo
            
        Returns:
            ClassificationModelMetrics: Objeto com as métricas carregadas ou None se arquivo não existir
        """
        metrics_dir = PathManager.get_path('metrics')
        path = metrics_dir / f"{stage_name}_metrics.json"

        if not path.exists():
            return None

        # Usar pandas para deserializar
        metrics_dict = pd.read_json(
            path, orient='index', typ='series').to_dict()
        return ClassificationModelMetrics.from_dict(metrics_dict)

    @staticmethod
    def get_all_metrics() -> pd.DataFrame:
        """Carrega todas as métricas salvas em um DataFrame."""
        metrics_dir = PathManager.get_path('output') / 'metrics'
        all_metrics = []

        for file in metrics_dir.glob('*_metrics.json'):
            with open(file) as f:
                metrics = json.load(f)
                all_metrics.append({
                    'stage_name': metrics['stage_name'],
                    'training_type': metrics['training_type'],
                    'cv_score': metrics['cv_score'],
                    **metrics['test_metrics']  # Expande métricas de teste
                })

        return pd.DataFrame(all_metrics)


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)
