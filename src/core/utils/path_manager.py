from pathlib import Path
import os
from typing import Dict, Optional, List


class PathManager:
    _instance = None
    _base_path: Optional[Path] = None
    _paths: Dict[str, Path] = {}
    _module_name: str = 'behavior'  # default module

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PathManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    # core/utils/path_manager.py


    def _find_project_root(self) -> Path:
        """Find project root by looking for project markers."""
        current_path = Path.cwd()

        # Lista de marcadores do projeto em ordem de prioridade
        root_markers = [
            'behavior-detection.code-workspace',  # Marcador específico do projeto
            'requirements.txt',
            'src',
            '.git'
        ]

        while True:
            # Verifica primeiro o marcador específico do projeto
            if (current_path / 'behavior-detection.code-workspace').exists():
                return current_path

            # Verifica outros marcadores
            for marker in root_markers[1:]:
                if (current_path / marker).exists():
                    # Validação adicional: verifica se é realmente o diretório behavior-detection
                    if current_path.name == 'behavior-detection':
                        return current_path
                    elif (current_path / 'behavior-detection').exists():
                        return current_path / 'behavior-detection'

            if current_path.parent == current_path:
                raise ValueError(
                    "Não foi possível encontrar o diretório raiz do projeto")

            current_path = current_path.parent


    def _initialize(self) -> None:
        """Initialize base path and standard project directories."""
        if self._base_path is None:
            self._base_path = self._find_project_root()

            # Setup standard project paths
            self._paths = {
                'root': self._base_path,
                'data': self._base_path / 'data',
                'output': self._base_path / 'output',  # Garante path consistente
                'models': self._base_path / 'output' / 'models',
                'src': self._base_path / 'src',
                'config': self._base_path / 'src' / self._module_name / 'config'
            }

            # Validação dos diretórios críticos
            required_dirs = ['data', 'src']
            for dir_name in required_dirs:
                if not self._paths[dir_name].exists():
                    raise ValueError(
                        f"Diretório {dir_name} não encontrado em {self._base_path}")

            # Criação dos diretórios de output
            for path in ['output', 'models']:
                self._paths[path].mkdir(parents=True, exist_ok=True)

    @classmethod
    def set_module(cls, module_name: str) -> None:
        """Set the current module name (e.g., 'behavior', 'mnist', 'emotion').
        
        Args:
            module_name: Name of the module to use
        """
        instance = cls()
        if module_name != instance._module_name:
            instance._module_name = module_name
            # Update config path for new module
            instance._paths['config'] = instance._paths['src'] / \
                module_name / 'config'
            instance._paths['config'].mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_path(cls, path_type: str = 'root') -> Path:
        """Get specified project path.
        
        Args:
            path_type: Type of path to return ('root', 'data', 'output', 'models', 'src', 'config')
            
        Returns:
            Path object for requested directory
        """
        instance = cls()
        if path_type not in instance._paths:
            raise ValueError(f"Invalid path type: {path_type}")
        return instance._paths[path_type]

    @classmethod
    def get_all_paths(cls) -> Dict[str, Path]:
        """Get dictionary of all project paths."""
        instance = cls()
        return instance._paths.copy()
