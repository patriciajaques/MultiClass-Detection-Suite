from behavior.behavior_model_params import BehaviorModelParams
from sklearn.pipeline import Pipeline
from core.feature_selection.lstm_sequence_processor import LSTMSequenceProcessor


class LSTMBehaviorModelParams(BehaviorModelParams):
    def __init__(self):
        # Primeiro chama o construtor da classe pai
        super().__init__()
        # Agora adiciona o modelo LSTM ao dicionário existente
        self.get_models()['LSTM'] = Pipeline(
            [('sequence_processor', LSTMSequenceProcessor())])

    def get_param_space(self, model_name: str) -> dict:
        if model_name == 'LSTM':
            return {
                'sequence_processor__sequence_length': [5, 10, 15],
                'sequence_processor__hidden_size': [32, 64, 128],
                'sequence_processor__num_layers': [1, 2],
                'sequence_processor__batch_size': [16, 32, 64],
                'sequence_processor__num_epochs': [10, 20],
                'sequence_processor__learning_rate': [0.001, 0.0001]
            }
        # Usa os parâmetros da classe pai para outros modelos
        return super().get_param_space(model_name)
