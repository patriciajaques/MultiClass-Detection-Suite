import os
import pickle
from datetime import datetime

class CheckpointManager:
    def __init__(self, base_path='drive/MyDrive/behavior_detection/checkpoints/'):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)  # Cria o diretório se não existir
        
    def save_checkpoint(self, state, filename):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        full_path = f"{self.base_path}{filename}_{timestamp}.pkl"
        
        with open(full_path, 'wb') as f:
            pickle.dump(state, f)
            
    def load_latest_checkpoint(self, filename_prefix):
        import glob
        files = glob.glob(f"{self.base_path}{filename_prefix}*.pkl")
        if not files:
            return None
        
        latest_file = max(files, key=os.path.getctime)
        with open(latest_file, 'rb') as f:
            return pickle.load(f)

    def merge_results(self, checkpoints):
        merged_models = {}
        for checkpoint in checkpoints:
            merged_models.update(checkpoint['trained_models'])
        return merged_models