from behavior.behavior_detection_pipeline import BehaviorDetectionPipeline
from behavior.behavior_data_loader import BehaviorDataLoader
from core.utils import file_utils
from core.utils.path_manager import PathManager


class EmotionDetectionPipeline(BehaviorDetectionPipeline):
    def __init__(self, target_column='estado_afetivo', n_iter=50, n_jobs=6, test_size=0.2):
        """
        Inicializa o pipeline de detecção de emoções.

        Args:
            target_column (str): Nome da coluna alvo
            n_iter (int): Número de iterações para otimização de hiperparâmetros
            n_jobs (int): Número de jobs paralelos para processamento
            test_size (float): Proporção dos dados para conjunto de teste

        Note:
            Esta classe estende BehaviorDetectionPipeline para realizar detecção
            de emoções ao invés de comportamentos, mantendo a mesma estrutura
            de pipeline e parâmetros.
        """
        super().__init__(
            target_column=target_column,
            n_iter=n_iter,
            n_jobs=n_jobs,
            test_size=test_size,
        )

    def load_and_clean_data(self):
        """Carrega e limpa o dataset focando nas emoções."""
        # Load data usando o loader existente
        data = BehaviorDataLoader.load_data(
            self.paths['data'] / 'new_logs_labels.csv', delimiter=';')
        print(f"Dataset inicial shape: {data.shape}")

        # Remove undefined emotions e NaN
        data = data[data[self.target_column].notna()]
        # data = self.data_cleaner.remove_instances_with_value(data, self.target_column, '?')

        # Unifica as classes "BOREDOM" e "FRUSTRATION" em "BORED_FRUSTRATION"
        data[self.target_column] = data[self.target_column].replace(
            {'BOREDOM': 'BORED_FRUSTRATION', 'FRUSTRATION': 'BORED_FRUSTRATION'})

        print(f"Classes de emoções: {data[self.target_column].unique()}")

        # Cria id único de sequencias (reuso do método da classe pai)
        data['sequence_id'] = self._create_sequence_ids(data)

        # Remove unnecessary columns usando configuração
        cleaned_data = self.data_cleaner.remove_columns(
            data, use_config=True)

        return cleaned_data


def main():
    """Main function to run the emotion detection pipeline."""

    PathManager.set_module('emotion')

    pipeline = EmotionDetectionPipeline(
        n_iter=50,
        n_jobs=6,
        test_size=0.2
    )
    pipeline.run()


if __name__ == "__main__":
    output_dir = "/Users/patricia/Documents/code/python-code/behavior-detection/output"
    file_utils.clear_output_directory(output_dir)
    main()
