import pandas as pd

from core.preprocessors.data_encoder import DataEncoder
from core.utils.path_manager import PathManager

class FeatureMappingReporter:
    def __init__(self):
        self.output_dir = PathManager.get_path('output')

    def log_feature_mappings(self, encoder: DataEncoder, filename_prefix="feature_mapping"):
        """Gera arquivo CSV com mapeamento das features categóricas"""
        mappings = encoder.get_feature_mapping()
        if not mappings:
            return

        filename = f"{filename_prefix}.csv"

        rows = []
        for feature, mapping in mappings.items():
            for code, value in mapping.items():
                rows.append({
                    'feature': feature,
                    'codigo': code,
                    'valor': value
                })

        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / filename, sep=';', index=False)
        print(f"Mapeamento das features salvo em: {filename}")

    def log_target_mappings(self, encoder: DataEncoder, filename_prefix="target_mapping"):
        """Gera arquivo CSV com mapeamento das classes target"""
        mappings = encoder.get_class_mapping()
        if not mappings:
            return

        filename = f"{filename_prefix}.csv"

        df = pd.DataFrame([
            {'codigo': code, 'classe': value}
            for code, value in mappings.items()
        ])
        df.to_csv(self.output_dir / filename, sep=';', index=False)
        print(f"Mapeamento das classes target salvo em: {filename}")
