from behavior.behavior_detection_pipeline import BehaviorDetectionPipeline
from behavior.data.behavior_data_loader import BehaviorDataLoader
from core.utils.path_manager import PathManager


class EmotionDetectionPipeline(BehaviorDetectionPipeline):
    def __init__(self, n_iter=50, n_jobs=6, test_size=0.2):
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
            target_column='estado_afetivo',
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

    def _verify_split_quality(self, train_data, test_data):
        """
        Sobrescreve o método de verificação de qualidade do split para trabalhar com strings de emoções.
        """
        # Verifica se todos os alunos estão em apenas um conjunto
        train_students = set(train_data['aluno'])
        test_students = set(test_data['aluno'])
        overlap = train_students & test_students
        assert len(
            overlap) == 0, f"Alunos presentes em ambos conjuntos: {overlap}"

        # Verifica proporções das classes com tolerância maior
        train_dist = train_data['comportamento'].value_counts(normalize=True)
        test_dist = test_data['comportamento'].value_counts(normalize=True)

        # Itera sobre todas as emoções presentes em ambos os conjuntos
        for emotion in set(train_dist.index) | set(test_dist.index):
            train_prop = train_dist.get(emotion, 0)
            test_prop = test_dist.get(emotion, 0)
            diff = abs(train_prop - test_prop)
            if diff >= 0.15:  # 15% de tolerância
                print(
                    f"Aviso: Diferença de {diff:.2%} na distribuição da emoção {emotion}")


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
    main()
