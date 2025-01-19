from pathlib import Path
import yaml
from typing import Dict, Any, Optional


class ConfigManager:
    """
    Gerenciador genérico de configurações do sistema.
    """

    def __init__(self, module_name: Optional[str] = None, config_path: Optional[Path] = None):
        """
        Args:
            module_name: Nome do módulo específico (ex: 'behavior', 'mnist')
            config_path: Caminho opcional para sobrescrever localização padrão
        """
        if config_path:
            self.config_root = config_path
        else:
            # Encontra a raiz do projeto
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent
            self.config_root = project_root / 'config'

        self.module_name = module_name
        self._config = {}
        self.load_configs()

    def load_configs(self) -> None:
        """Carrega configurações compartilhadas e específicas do módulo."""
        try:
            # Carrega arquivos da pasta raiz de config
            self._load_from_directory(self.config_root)

            # Carrega configurações específicas do módulo se especificado
            if self.module_name:
                module_config_dir = self.config_root / self.module_name
                if module_config_dir.exists():
                    self._load_from_directory(module_config_dir)

        except Exception as e:
            raise RuntimeError(f"Erro ao carregar configurações: {str(e)}")

    def _load_from_directory(self, directory: Path) -> None:
        """Carrega todos os arquivos YAML de um diretório."""
        if directory.exists():
            for config_file in directory.glob('*.yaml'):
                with open(config_file, 'r') as f:
                    self._config.update(yaml.safe_load(f))

    def get_config(self, config_path: str, default: Any = None) -> Any:
        """
        Obtém valor de configuração usando notação de ponto.
        
        Args:
            config_path: Caminho da configuração (ex: 'models.available_models')
            default: Valor padrão se configuração não encontrada
            
        Returns:
            Valor da configuração ou default
        """
        current = self._config
        for key in config_path.split('.'):
            if isinstance(current, dict):
                current = current.get(key, default)
            else:
                return default
        return current

    def get_all_configs(self) -> Dict:
        """Retorna todas as configurações carregadas."""
        return self._config
