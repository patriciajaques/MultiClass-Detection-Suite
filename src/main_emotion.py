from behavior.behavior_detection_pipeline import BehaviorDetectionPipeline
from behavior.data.behavior_data_loader import BehaviorDataLoader
from core.preprocessors.data_cleaner import DataCleaner
import pandas as pd


class EmotionDetectionPipeline(BehaviorDetectionPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_and_clean_data(self):
        """Carrega e limpa o dataset focando nas emoções."""
        # Load data usando o loader existente
        data = BehaviorDataLoader.load_data(
            self.paths['data'] / 'new_logs_labels.csv', delimiter=';')
        print(f"Dataset inicial shape: {data.shape}")

        # Remove undefined emotions
        data = self.data_cleaner.remove_instances_with_value(
            data, 'estado_afetivo', '?')

        print(f"Classes de emoções: {data['estado_afetivo'].unique()}")

        # Cria id único de sequencias (reuso do método da classe pai)
        data['sequence_id'] = self._create_sequence_ids(data)

        # Remove unnecessary columns usando configuração
        cleaned_data = self.data_cleaner.remove_columns(
            data, use_config=True)

        return cleaned_data

    def prepare_data(self, data):
        """Prepara os dados para treinamento focando em emoções."""
        print(f"\nIniciando preparação dos dados para detecção de emoções...")

        # Validação inicial das colunas necessárias
        self._validate_split_columns(data)

        # Substituir 'comportamento' por 'estado_afetivo' como target
        data['target'] = data['estado_afetivo']

        # Reutilizar o método prepare_data da classe pai
        return super().prepare_data(data)


def main():
    """Main function to run the emotion detection pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Execute emotion detection pipeline for specific stages')
    parser.add_argument('--start-stage', type=int,
                        help='Starting stage number')
    parser.add_argument('--end-stage', type=int, help='Ending stage number')
    args = parser.parse_args()

    stage_range = None
    if args.start_stage is not None and args.end_stage is not None:
        stage_range = (args.start_stage, args.end_stage)

    pipeline = EmotionDetectionPipeline(
        n_iter=50,
        n_jobs=6,
        test_size=0.2,
        stage_range=stage_range
    )
    pipeline.run()


if __name__ == "__main__":
    main()
