from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TemporalFeaturesProcessor(BaseEstimator, TransformerMixin):
    """
    Processador ajustado para as características específicas dos logs do sistema tutor.
    """

    def __init__(self,
                 sequence_windows: List[int] = [3, 5, 10],
                 include_last_n_steps: int = 3):
        """
        Args:
            sequence_windows: Tamanhos das janelas de sequência para agregação
            include_last_n_steps: Número de passos anteriores a incluir
        """
        self.sequence_windows = sequence_windows
        self.include_last_n_steps = include_last_n_steps

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma o DataFrame adicionando features temporais específicas para o sistema tutor.
        """
        df = X.copy()

        # Ordenar por aluno e sequência de log
        df = df.sort_values(['aluno', 'num_dia', 'num_log'])

        # 1. Features de sequência de passos
        self._add_step_sequence_features(df)

        # 2. Features de tempo e eficiência
        self._add_time_efficiency_features(df)

        # 3. Features de padrões de comportamento
        self._add_behavior_pattern_features(df)

        # 4. Features de progresso na equação
        self._add_equation_progress_features(df)

        # 5. Validar e limpar features
        df = self._validate_and_clean_features(df)

        return df

    def _add_step_sequence_features(self, df: pd.DataFrame) -> None:
        """Adiciona features relacionadas à sequência de passos na resolução."""
        # Passos anteriores
        for i in range(1, self.include_last_n_steps + 1):
            df[f'prev_step_correct_{i}'] = df.groupby(
                'aluno')['ultimo_passo_correto'].shift(i)
            df[f'prev_log_type_{i}'] = df.groupby('aluno')['log_type'].shift(i)
            df[f'prev_eq_step_verification_{i}'] = (df.groupby('aluno')['log_type']
                                                    .shift(i)
                                                    .eq('step_verification')
                                                    .astype(int))

        # Taxa de acertos nos últimos N passos
        for window in self.sequence_windows:
            df[f'success_rate_{window}'] = (df.groupby('aluno')['ultimo_passo_correto']
                                            .rolling(window=window)
                                            .mean()
                                            .reset_index(0, drop=True))

    def _add_time_efficiency_features(self, df: pd.DataFrame) -> None:
        """Adiciona features relacionadas ao tempo e eficiência."""
        # Tempo relativo comparado à média
        df['time_vs_mean'] = df['tempo_equacao'] / df['tempo_medio_eq_diario']

        # Eficiência do passo
        df['step_efficiency'] = df['tempo_passo'] / \
            df['tempo_medio_passo_diario']

        # Tempo ocioso relativo
        df['relative_idle_time'] = df['idle_time_acumulado'] / df['tempo_equacao']

        # Velocidade de resolução
        for window in self.sequence_windows:
            df[f'avg_step_time_{window}'] = (df.groupby('aluno')['tempo_passo']
                                             .rolling(window=window)
                                             .mean()
                                             .reset_index(0, drop=True))

    def _add_behavior_pattern_features(self, df: pd.DataFrame) -> None:
        """Adiciona features relacionadas a padrões de comportamento."""
        # Mudanças de comportamento
        df['behavior_changed'] = (df.groupby('aluno')['comportamento']
                                  .shift() != df['comportamento']).astype(int)

        # Contagem de verificações de passo
        df['step_verifications'] = (
            df['log_type'] == 'step_verification').astype(int)

        # Padrões de verificação
        for window in self.sequence_windows:
            df[f'verification_rate_{window}'] = (df.groupby('aluno')['step_verifications']
                                                 .rolling(window=window)
                                                 .mean()
                                                 .reset_index(0, drop=True))

        # Tempo entre verificações
        df['time_between_verifications'] = (df[df['log_type'] == 'step_verification']
                                            .groupby('aluno')['tempo_passo']
                                            .diff())

    def _add_equation_progress_features(self, df: pd.DataFrame) -> None:
        """Adiciona features relacionadas ao progresso na equação."""
        # Progresso na equação atual
        df['steps_in_equation'] = df.groupby(['aluno', 'num_dia'])[
            'num_log'].transform('count')
        df['step_position'] = df.groupby(['aluno', 'num_dia'])[
            'num_log'].transform(lambda x: (x - x.min() + 1))
        df['relative_position'] = df['step_position'] / df['steps_in_equation']

        # Efetividade no progresso
        df['progress_effectiveness'] = df['num_passos_corretos_eq'] / \
            df['num_passos_equacao']

        # Taxa de erro
        df['error_rate'] = 1 - df['efetividade_passos_eq']

    def get_feature_names(self) -> List[str]:
        """Retorna os nomes das features temporais criadas."""
        feature_names = []

        # Features de sequência
        for i in range(1, self.include_last_n_steps + 1):
            feature_names.extend([
                f'prev_step_correct_{i}',
                f'prev_log_type_{i}',
                f'prev_eq_step_verification_{i}'
            ])

        # Features de janela
        for window in self.sequence_windows:
            feature_names.extend([
                f'success_rate_{window}',
                f'avg_step_time_{window}',
                f'verification_rate_{window}'
            ])

        # Features de tempo e eficiência
        feature_names.extend([
            'time_vs_mean',
            'step_efficiency',
            'relative_idle_time'
        ])

        # Features de comportamento
        feature_names.extend([
            'behavior_changed',
            'step_verifications',
            'time_between_verifications'
        ])

        # Features de progresso
        feature_names.extend([
            'steps_in_equation',
            'step_position',
            'relative_position',
            'progress_effectiveness',
            'error_rate'
        ])

        return feature_names


    def _validate_and_clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validação e limpeza completa das features temporais"""
        df_clean = df.copy()

        # Tratar divisões por zero
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            df_clean[col] = self._safe_division(df_clean[col], 1)

        # Remover infinitos
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

        # Limitar valores extremos
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].clip(-1e308, 1e308)

        return df_clean

    def _safe_division(self, a, b, fill_value=0):
        """Realiza divisão segura evitando divisões por zero."""
        if isinstance(b, (int, float)):
            mask = b != 0
        else:
            mask = b.astype(bool)
        return np.where(mask, a / np.where(mask, b, 1), fill_value)
