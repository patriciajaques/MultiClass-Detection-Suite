from pathlib import Path
import yaml
from typing import Dict, Any, Optional


class ConfigManager:
    """
    Gerenciador genérico de configurações do sistema.
    """

    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self._config = []  # Inicializa como uma lista vazia
        self.load_configs()

    def load_configs(self) -> None:
        """Carrega configurações compartilhadas e específicas do módulo."""
        try:
            # Carrega arquivos da pasta raiz de config
            self._load_from_directory(self.config_dir)

        except Exception as e:
            raise RuntimeError(f"Erro ao carregar configurações: {str(e)}")

    def _load_from_directory(self, directory: Path) -> None:
        """Carrega todos os arquivos YAML de um diretório."""

        if not self.config_dir.exists():
            raise ValueError(
                f"Diretório de configuração não encontrado: {self.config_dir}")

        for config_file in directory.glob('*.yaml'):
            with open(config_file, 'r') as f:
                self._config.append(yaml.safe_load(f))


    def get_config(self, config_key: str) -> dict:
        """
        Retorna configuração específica com fallback para configurações padrão
        
        Args:
            config_key (str): Chave da configuração desejada
            
        Returns:
            dict: Configuração solicitada ou configuração padrão
        """
        try:
            # Tenta encontrar a configuração nos arquivos carregados
            for config_dict in self._config:
                if config_key in config_dict:
                    return config_dict[config_key]

            # Se não encontrar, retorna configuração padrão
            return self._get_default_config(config_key)
        except Exception as e:
            print(f"Aviso: Erro ao carregar configuração '{config_key}': {str(e)}")
            return self._get_default_config(config_key)


    def _get_default_config(self, config_key: str) -> dict:
        """
        Retorna configurações padrão para diferentes chaves
        """
        defaults = {
            'training_settings': {
                'models': ['Naive Bayes'],  # Começamos com modelo mais simples
                'selectors': ['none']       # Sem seleção de features inicialmente
            },
            'columns_to_remove': {
                'redundant': ['id_log'],
                'irrelevant': ['grupo']
            }
        }

        if config_key in defaults:
            print(f"Usando configuração padrão para '{config_key}'")
            return defaults[config_key]

        raise KeyError(
            f"Configuração '{config_key}' não encontrada e não há padrão definido")

    def get_all_configs(self) -> Dict:
        """Retorna todas as configurações carregadas."""
        return self._config
