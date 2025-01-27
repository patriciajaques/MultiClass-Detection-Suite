from pathlib import Path
from datetime import datetime
import pandas as pd


class FeatureMappingLogger:
   def __init__(self, output_dir="../output/mappings/"):
       self.output_dir = Path(output_dir)
       self.output_dir.mkdir(parents=True, exist_ok=True)

   def log_feature_mappings(self, encoder, filename_prefix="feature_mapping"):
       """Gera arquivo CSV com mapeamento das features categóricas"""
       mappings = encoder.get_feature_mapping()
       if not mappings:
           return

       timestamp = datetime.now().strftime("%Y%m%d_%H%M")
       filename = f"{filename_prefix}_{timestamp}.csv"

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

   def log_target_mappings(self, encoder, filename_prefix="target_mapping"):
       """Gera arquivo CSV com mapeamento das classes target"""
       mappings = encoder.get_class_mapping()
       if not mappings:
           return

       timestamp = datetime.now().strftime("%Y%m%d_%H%M")
       filename = f"{filename_prefix}_{timestamp}.csv"

       df = pd.DataFrame([
           {'codigo': code, 'classe': value}
           for code, value in mappings.items()
       ])
       df.to_csv(self.output_dir / filename, sep=';', index=False)
       print(f"Mapeamento das classes target salvo em: {filename}")
