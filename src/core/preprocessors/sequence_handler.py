import numpy as np
import pandas as pd


class SequenceHandler:
   """
   Gerencia sequências temporais para análise de comportamento em sistemas tutores.
   
   Esta classe cria e mantém um mapeamento consistente de sequências temporais,
   permitindo identificar padrões sequenciais nos dados de logs do estudante.
   
   Attributes:
       sequence_mapping (dict): Dicionário que mapeia sequências brutas para IDs numéricos
       next_id (int): Próximo ID disponível para novas sequências
   
   Example:
       >>> handler = SequenceHandler()
       >>> handler.fit(train_data)
       >>> sequence_ids = handler.transform(test_data)
   """

   def __init__(self):
       """Inicializa o mapeamento de sequências vazio."""
       self.sequence_mapping = {}
       self.next_id = 0

   def fit(self, X: pd.DataFrame):
       """
       Cria mapeamento inicial de sequências do conjunto de treino.

       Args:
           X (pd.DataFrame): DataFrame contendo colunas grupo, aluno, num_dia, num_log

       Returns:
           self: Retorna a instância para permitir encadeamento
       """
       sequences = self._create_raw_sequences(X)
       for seq in sequences:
           if seq not in self.sequence_mapping:
               self.sequence_mapping[seq] = self.next_id
               self.next_id += 1
       return self

   def transform(self, X: pd.DataFrame) -> np.ndarray:
       """
       Transforma sequências em IDs numéricos, marcando sequências não vistas como -1.

       Args:
           X (pd.DataFrame): DataFrame contendo colunas grupo, aluno, num_dia, num_log

       Returns:
           np.ndarray: Array com IDs numéricos das sequências
       """
       sequences = self._create_raw_sequences(X)
       return np.array([self.sequence_mapping.get(seq, -1) for seq in sequences])

   def _create_raw_sequences(self, X: pd.DataFrame) -> np.ndarray:
       """
       Cria identificadores de sequência concatenando informações temporais.

       Args:
           X (pd.DataFrame): DataFrame contendo colunas grupo, aluno, num_dia, num_log

       Returns:
           np.ndarray: Array com sequências brutas no formato grupo_aluno_dia_log
       """
       return (X['grupo'].astype(str) + '_' +
               X['aluno'].astype(str) + '_' +
               X['num_dia'].astype(str) + '_' +
               X['num_log'].astype(str))
