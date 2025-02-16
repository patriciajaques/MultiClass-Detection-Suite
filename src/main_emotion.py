import os
from behavior.behavior_detection_pipeline import BehaviorDetectionPipeline
from core.preprocessors.data_loader import DataLoader
from core.utils import file_utils
from core.utils.path_manager import PathManager


class EmotionDetectionPipeline(BehaviorDetectionPipeline):
    def __init__(self, target_column='estado_afetivo', n_iter=50, n_jobs=6,
                 val_size=0.25, test_size=0.2, group_feature=None,
                 training_strategy_name='optuna', use_voting_classifier=True):
        """
        Inicializa o pipeline de detecção de emoções.
        
        Args:
            target_column: Nome da coluna alvo (default: 'estado_afetivo')
            n_iter: Número de iterações para otimização
            n_jobs: Número de jobs paralelos
            val_size: Tamanho do conjunto de validação
            test_size: Tamanho do conjunto de teste
            training_strategy_name: Estratégia de treinamento ('optuna', 'grid', 'random')
            use_voting_classifier: Se deve usar classificador por votação
        """
        super().__init__(
            target_column=target_column,
            n_iter=n_iter,
            n_jobs=n_jobs,
            val_size=val_size,
            test_size=test_size,
            group_feature=group_feature,
            training_strategy_name=training_strategy_name,
            use_voting_classifier=use_voting_classifier
        )

    def clean_data(self, data):

        # Remove undefined emotions e NaN
        data = data[data[self.target_column].notna()]

        # Unifica as classes "BOREDOM" e "FRUSTRATION" em "BORED_FRUSTRATION"
        # data[self.target_column] = data[self.target_column].replace(
        #     {'BOREDOM': 'BORED_FRUSTRATION', 'FRUSTRATION': 'BORED_FRUSTRATION'})

        self.logger.info(
            f"Classes de emoções: {data[self.target_column].unique()}")

        # Remove unnecessary columns
        columns_to_keep = ['aluno', 'num_dia',
                           'num_log', 'sequence_id', self.target_column]
        columns_to_remove = self.data_cleaner.get_columns_to_remove(
            self.config_manager)
        cleaned_data = self.data_cleaner.clean_data(
            data,
            target_column=self.target_column,
            undefined_value='?',
            columns_to_remove=columns_to_remove,
            columns_to_keep=columns_to_keep,
            handle_multicollinearity=True
        )

        return cleaned_data


def main():
    """Main function to run the emotion detection pipeline."""

    PathManager.set_module('emotion')

    pipeline = EmotionDetectionPipeline(
        target_column='estado_afetivo',
        n_iter=50,
        n_jobs=4,
        val_size=None,
        test_size=0.2,
        group_feature=None,
        # pode ser 'optuna' (default), 'grid' ou 'random'
        training_strategy_name='optuna',
        use_voting_classifier=True
    )
    pipeline.run()


if __name__ == "__main__":
    os.system('clear')
    output_dir = "/Users/patricia/Documents/code/python-code/behavior-detection/output"
    file_utils.clear_output_directory(output_dir)
    main()
