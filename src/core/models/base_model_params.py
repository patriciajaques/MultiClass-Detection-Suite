from abc import ABC, abstractmethod
from typing import Any, Dict, List

from sklearn.base import BaseEstimator

class BaseModelParams(ABC):
    """
    Classe base abstrata que define a interface para todos os parâmetros de modelo.
    """
    @abstractmethod
    def get_models(self) -> Dict[str, BaseEstimator]:
        """Retorna o dicionário de modelos base."""
        pass
    
    @abstractmethod
    def get_param_space(self, model_name: str) -> Dict[str, Any]:
        """Retorna o espaço de parâmetros para um modelo específico."""
        pass
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Retorna lista de modelos disponíveis."""
        return list(cls().get_models().keys())
