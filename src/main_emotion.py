from behavior.behavior_detection_pipeline import BehaviorDetectionPipeline
from behavior.data.behavior_data_loader import BehaviorDataLoader


class EmotionDetectionPipeline(BehaviorDetectionPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_and_clean_data(self):
        """Carrega e limpa o dataset focando nas emoções."""
        # Load data usando o loader existente
        data = BehaviorDataLoader.load_data(
            self.paths['data'] / 'new_logs_labels.csv', delimiter=';')
        print(f"Dataset inicial shape: {data.shape}")

        # Remove undefined emotions e NaN
        data = data[data['estado_afetivo'].notna()]
        # data = self.data_cleaner.remove_instances_with_value(data, 'estado_afetivo', '?')
        
        # Unifica as classes "BOREDOM" e "FRUSTRATION" em "BORED_FRUSTRATION"
        data['estado_afetivo'] = data['estado_afetivo'].replace(
            {'BOREDOM': 'BORED_FRUSTRATION', 'FRUSTRATION': 'BORED_FRUSTRATION'})

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

        # Converte o target para estado_afetivo para poder aproveitar o método de preparação da classe pai
        data = data.copy()
        data['comportamento'] = data['estado_afetivo']
        # Remove a coluna 'estado_afetivo' do dataframe após fazer a cópia
        data = data.drop(columns=['estado_afetivo'])

        print(
            f"Distribuição de emoções:\n{data['comportamento'].value_counts()}")

        # Reutiliza o método prepare_data da classe pai
        return super().prepare_data(data)
    

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
