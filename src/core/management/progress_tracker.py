import json

from core.utils.path_manager import PathManager


class ProgressTracker:
    def __init__(self):
       self.progress_file = PathManager.get_path('output') / 'progress.json'
       self.completed_pairs = self._load_progress()

    def _load_progress(self):
        try:
            with open(self.progress_file, 'r') as f:
                return set(json.load(f)['completed_pairs'])
        except FileNotFoundError:
            return set()

    def save_progress(self, model_selector_pair):
        self.completed_pairs.add(model_selector_pair)
        with open(self.progress_file, 'w') as f:
            json.dump({'completed_pairs': list(self.completed_pairs)}, f)

    def is_completed(self, model_selector_pair):
        return model_selector_pair in self.completed_pairs
