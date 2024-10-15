import pandas as pd
from sklearn.preprocessing import LabelEncoder
from core.preprocessors.column_selector import ColumnSelector
from core.preprocessors.data_encoder import DataEncoder

class BehaviorDataEncoder(DataEncoder):
    def __init__(self, num_classes=5):
        super().__init__(num_classes)

    @staticmethod
    def encode_y(y):
        y_encoded = LabelEncoder().fit_transform(y)
        return y_encoded

    # def select_columns(self, X: pd.DataFrame):
    #     """ Seleciona colunas numéricas, nominais e ordinais
    #     para o conjunto de dados de comportamento.
    #     """
    #     self.column_selector = ColumnSelector(X, self.num_classes)
    #     behavior_columns = self.column_selector.get_columns_by_regex('^comportamento|^ultimo_comportamento')
    #     personality_columns = self.column_selector.get_columns_by_regex('^traco_')
    #     personality_columns = [col for col in personality_columns if not col.endswith('_cat')]
    #     all_columns = set(X.columns)
    #     selected_columns = set(behavior_columns + personality_columns)
    #     self.numerical_columns = list(all_columns - selected_columns)
    #     self.nominal_columns = behavior_columns + personality_columns
    #     self.ordinal_columns = []
    #     self.ordinal_categories = {}
