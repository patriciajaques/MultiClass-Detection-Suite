import pickle
from datetime import datetime
from pathlib import Path
from core.utils.path_manager import PathManager


class CheckpointManager:
    """
    Gerencia o salvamento e carregamento de checkpoints do modelo.
    Checkpoints são salvos em output/checkpoints/{module_name}/
    """

    def __init__(self, checkpoint_dir: str = None):
        """
        Inicializa o gerenciador de checkpoints.
        
        Args:
            checkpoint_dir: Diretório opcional para checkpoints.
                          Se None, usa output/checkpoints/{module_name}
        """
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            # Cria um subdiretório 'checkpoints' dentro do diretório output
            self.checkpoint_dir = PathManager.get_path(
                'output') / 'checkpoints'

        # Cria o diretório se não existir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, state, filename: str):
        """
        Salva um checkpoint com timestamp.
        
        Args:
            state: Estado do modelo a ser salvo
            filename: Nome base do arquivo
        
        Returns:
            str: Caminho completo do arquivo salvo
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        full_path = self.checkpoint_dir / f"{filename}_{timestamp}.pkl"

        with open(full_path, 'wb') as f:
            pickle.dump(state, f)

        return str(full_path)

    def load_latest_checkpoint(self, filename_prefix: str):
        """
        Carrega o checkpoint mais recente com o prefixo especificado.
        
        Args:
            filename_prefix: Prefixo do arquivo a ser carregado
            
        Returns:
            O estado do modelo carregado ou None se não encontrar checkpoint
        """
        pattern = f"{filename_prefix}*.pkl"
        checkpoint_files = list(self.checkpoint_dir.glob(pattern))

        if not checkpoint_files:
            return None

        latest_file = max(checkpoint_files, key=lambda x: x.stat().st_mtime)

        with open(latest_file, 'rb') as f:
            return pickle.load(f)

    def merge_results(self, checkpoints):
        """
        Combina resultados de múltiplos checkpoints.
        
        Args:
            checkpoints: Lista de checkpoints a serem combinados
            
        Returns:
            dict: Modelos combinados dos checkpoints
        """
        merged_models = {}
        for checkpoint in checkpoints:
            if isinstance(checkpoint, dict) and 'trained_models' in checkpoint:
                merged_models.update(checkpoint['trained_models'])
        return merged_models

    def cleanup_old_checkpoints(self, max_checkpoints: int = 5):
        """
        Remove checkpoints antigos mantendo apenas os mais recentes.
        
        Args:
            max_checkpoints: Número máximo de checkpoints a manter
        """
        checkpoint_files = list(self.checkpoint_dir.glob("*.pkl"))
        if len(checkpoint_files) <= max_checkpoints:
            return

        # Ordena por data de modificação
        sorted_files = sorted(checkpoint_files,
                              key=lambda x: x.stat().st_mtime,
                              reverse=True)

        # Remove os mais antigos
        for file in sorted_files[max_checkpoints:]:
            try:
                file.unlink()
            except Exception as e:
                print(f"Erro ao remover checkpoint antigo {file}: {e}")
