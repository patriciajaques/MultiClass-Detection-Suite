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

    def _find_project_root(self) -> Path:
        """Find project root by looking for common project markers."""
        current_path = Path.cwd()
        print(f"Starting search at: {current_path}")

        root_markers = [
            'pyproject.toml',
            'setup.py',
            'src',
            '.git'
        ]

        while True:
            print(f"Checking directory: {current_path}")

            # Lista quais marcadores foram encontrados
            found_markers = [marker for marker in root_markers if (
                current_path / marker).exists()]
            if found_markers:
                print(f"Found markers: {found_markers}")
                return current_path

            if current_path.parent == current_path:
                print("Reached filesystem root")
                break

            current_path = current_path.parent
            print(f"Moving up to: {current_path}")

        print(f"No markers found, using current directory: {Path.cwd()}")
        return Path.cwd()

    def _initialize(self) -> None:
        """Initialize base path and standard project directories."""
        if self._base_path is None:
            self._base_path = self._find_project_root()

            # Setup standard project paths
            self._paths = {
                'root': self._base_path,
                'data': self._base_path / 'data',
                'output': self._base_path / 'output',
                'models': self._base_path / 'models',
                'src': self._base_path / 'src',
                'config': self._base_path / 'src' / self._module_name / 'config'
            }

            # Create directories if they don't exist
            for path in self._paths.values():
                path.mkdir(parents=True, exist_ok=True)

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
