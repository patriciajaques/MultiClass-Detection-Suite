import pandas as pd
from core.preprocessors.base_feature_engineer import BaseFeatureEngineer


class BehaviorFeatureEngineer(BaseFeatureEngineer):
    """Feature engineering específico para detecção de comportamentos."""

    def __init__(self):
        super().__init__()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Reseta o conjunto de colunas a remover a cada transformação
        self.columns_to_drop = set()
        df_transformed = df.copy()

        df_transformed = self._compute_relative_velocity(df_transformed, replace=True)
        df_transformed = self._create_verification_status(df_transformed, replace=True)
        df_transformed = self._create_accuracy_eq(df_transformed, replace=True)
        df_transformed = self._create_click_ratio(df_transformed, replace=True)
        df_transformed = self._create_operation_complexity(df_transformed, replace=True)
        df_transformed = self._create_aggregated_expert_by_complexity(df_transformed, replace=True)
        df_transformed = self._create_window_active(df_transformed, replace=True)
        df_transformed = self._create_session_position(df_transformed, replace=False)
        df_transformed = self._create_engagement_index(df_transformed, replace=True)

        # Após todas as transformações, remove as colunas acumuladas
        df_transformed = df_transformed.drop(columns=list(self.columns_to_drop), errors='ignore')
        return df_transformed

    def get_feature_names(self) -> list:
        return self.new_feature_names

    def _transform_features(self,
                            df: pd.DataFrame,
                            required_columns: list[str],
                            new_column_name: str,
                            transform_func: callable,
                            replace: bool = True) -> pd.DataFrame:
        """
        Método template para transformação de features.
        
        Cria a nova feature e, se replace=True, acumula as colunas que deverão ser removidas
        ao final de todas as transformações.
        
        Args:
            df (pd.DataFrame): DataFrame de entrada
            required_columns (list[str]): Colunas necessárias para a transformação
            new_column_name (str): Nome da nova feature a ser criada
            transform_func (callable): Função que cria a nova feature (deve receber o df e retornar uma Series/Array)
            replace (bool): Se True, as colunas originais serão removidas somente ao final
            
        Returns:
            pd.DataFrame: DataFrame transformado
        """
        # Validação de entrada
        if not all(col in df.columns for col in required_columns):
            raise ValueError(
                f"DataFrame must contain columns: {required_columns}")

        # Cria uma cópia para não alterar o original
        df_transformed = df.copy()

        # Aplica a transformação e cria a nova coluna
        df_transformed[new_column_name] = transform_func(df)

        # Se replace=True, acumula as colunas a serem removidas (mas não remove de imediato)
        if replace:
            self.columns_to_drop.update(required_columns)

        return df_transformed

    def _compute_relative_velocity(self, df: pd.DataFrame, replace: bool = True) -> pd.DataFrame:
        """
        Computa a velocidade relativa entre 'tempo_equacao' e 'tempo_medio_eq_diario'.
        """
        epsilon = 1e-6
        return self._transform_features(
            df=df,
            required_columns=['tempo_equacao', 'tempo_medio_eq_diario'],
            new_column_name='velocidade_relativa_equacao',
            transform_func=lambda x: x['tempo_equacao'] / (x['tempo_medio_eq_diario']+epsilon),
            replace=replace
        )

    def _create_verification_status(self, df: pd.DataFrame, replace: bool = True) -> pd.DataFrame:
        """
        Cria a feature binária 'verified' combinando 'verificado_com_mouse' e 'verificado_com_teclado'.
        """
        return self._transform_features(
            df=df,
            required_columns=['verificado_com_mouse',
                              'verificado_com_teclado'],
            new_column_name='verified',
            transform_func=lambda x: x['verificado_com_mouse'] | x['verificado_com_teclado'],
            replace=replace
        )

    def _create_accuracy_eq(self, df: pd.DataFrame, replace: bool = True) -> pd.DataFrame:
        """
        Cria a feature 'accuracy_eq' como razão entre passos corretos e total de passos da equação.
        """
        return self._transform_features(
            df=df,
            required_columns=['num_passos_corretos_eq', 'num_passos_equacao'],
            new_column_name='accuracy_eq',
            transform_func=lambda x: x['num_passos_corretos_eq'] /
            (x['num_passos_equacao'] + 1),
            replace=replace
        )

    def _create_click_ratio(self, df: pd.DataFrame, replace: bool = True) -> pd.DataFrame:
        """
        Cria a feature 'click_ratio' como a razão entre o número de cliques no passo atual e o acumulado.
        """
        return self._transform_features(
            df=df,
            required_columns=['num_click_passo', 'num_click_acumulado'],
            new_column_name='click_ratio',
            transform_func=lambda x: x['num_click_passo'] /
            (x['num_click_acumulado'] + 1),
            replace=replace
        )

    def _create_operation_complexity(self, df: pd.DataFrame, replace: bool = True) -> pd.DataFrame:
        """
        Agrupa operações em três categorias: básicas, frações e avançadas.
        """
        # Operações básicas
        basic_cols = ['eq_AD', 'eq_SB', 'eq_MT', 'eq_DV']
        df = self._transform_features(
            df=df,
            required_columns=basic_cols,
            new_column_name='basic_operations',
            transform_func=lambda x: x['eq_AD'] +
            x['eq_SB'] + x['eq_MT'] + x['eq_DV'],
            replace=replace
        )

        # Operações com frações
        fraction_cols = ['eq_AF', 'eq_MF']
        df = self._transform_features(
            df=df,
            required_columns=fraction_cols,
            new_column_name='fraction_operations',
            transform_func=lambda x: x['eq_AF'] + x['eq_MF'],
            replace=replace
        )

        # Operações avançadas
        advanced_cols = ['eq_DM', 'eq_MM', 'eq_OI', 'eq_SP']
        df = self._transform_features(
            df=df,
            required_columns=advanced_cols,
            new_column_name='advanced_operations',
            transform_func=lambda x: x['eq_DM'] +
            x['eq_MM'] + x['eq_OI'] + x['eq_SP'],
            replace=replace
        )

        return df

    def _create_aggregated_expert_by_complexity(self, df: pd.DataFrame, replace: bool = True) -> pd.DataFrame:
        """
        Agrega as expert features agrupando-as por janela temporal e complexidade da operação.
        """
        # Agrupamento para janela Clip
        clip_basic = ['eq_MT_fator_clip']
        clip_advanced = ['eq_OI_fator_clip']
        if clip_basic:
            df = self._transform_features(
                df=df,
                required_columns=clip_basic,
                new_column_name='clip_basic_expert',
                transform_func=lambda x: x[clip_basic].sum(axis=1),
                replace=replace
            )
        if clip_advanced:
            df = self._transform_features(
                df=df,
                required_columns=clip_advanced,
                new_column_name='clip_advanced_expert',
                transform_func=lambda x: x[clip_advanced].sum(axis=1),
                replace=replace
            )

        # Agrupamento para janela Diário
        diario_basic = ['misc_EqPrim_Ad_Sin_diario',
                        'misc_EqPrim_Dv_Sin_diario', 'eq_SB_erro_diario']
        diario_advanced = ['eq_MM_acerto_diario', 'eq_DM_erro_diario',
                           'eq_DM_fator_diario', 'eq_MM_fator_diario',
                           'misc_OI_Dv_Plus_Dv_Minus_diario', 'misc_OI_Dv_Plus_Dv_Plus_diario',
                           'misc_OI_Dv_Minus_Dv_Plus_diario']
        if diario_basic:
            df = self._transform_features(
                df=df,
                required_columns=diario_basic,
                new_column_name='diario_basic_expert',
                transform_func=lambda x: x[diario_basic].sum(axis=1),
                replace=replace
            )
        if diario_advanced:
            df = self._transform_features(
                df=df,
                required_columns=diario_advanced,
                new_column_name='diario_advanced_expert',
                transform_func=lambda x: x[diario_advanced].sum(axis=1),
                replace=replace
            )

        # Agrupamento para janela Total
        total_basic = ['misc_EqPrim_Ad_Inc_total', 'misc_EqPrim_Dv_Inc_total']
        total_fraction = ['eq_AF_acerto_total',
                          'eq_AF_fator_total', 'eq_MF_fator_total']
        total_advanced = ['eq_DM_erro_total', 'eq_MM_fator_total', 'eq_SP',
                          'misc_OI_Dv_Plus_Dv_Minus_total', 'misc_OI_Mt_Minus_Ad_total',
                          'misc_OI_Dv_Plus_Dv_Plus_total', 'misc_OI_Mt_Plus_Ad_total',
                          'misc_OI_Dv_Minus_Dv_Plus_total', 'misc_OI_Mt_Plus_Sb_total', 'misc_OI_Dv_Plus_Sb_total']
        if total_basic:
            df = self._transform_features(
                df=df,
                required_columns=total_basic,
                new_column_name='total_basic_expert',
                transform_func=lambda x: x[total_basic].sum(axis=1),
                replace=replace
            )
        if total_fraction:
            df = self._transform_features(
                df=df,
                required_columns=total_fraction,
                new_column_name='total_fraction_expert',
                transform_func=lambda x: x[total_fraction].sum(axis=1),
                replace=replace
            )
        if total_advanced:
            df = self._transform_features(
                df=df,
                required_columns=total_advanced,
                new_column_name='total_advanced_expert',
                transform_func=lambda x: x[total_advanced].sum(axis=1),
                replace=replace
            )

        return df

    def _create_window_active(self, df: pd.DataFrame, replace: bool = True) -> pd.DataFrame:
        """
        Cria a feature binária 'window_active' baseada em eventos de foco/perda de foco da janela.
        """
        def compute_window_active(row):
            if row['type_window_lost_focus'] == 1 or row['type_left_window'] == 1:
                return 0
            if row['type_window_gained_focus'] == 1 or row['type_entered_window'] == 1:
                return 1
            return 0

        return self._transform_features(
            df=df,
            required_columns=[
                'type_window_gained_focus', 'type_entered_window',
                'type_window_lost_focus', 'type_left_window'
            ],
            new_column_name='window_active',
            transform_func=lambda x: x.apply(compute_window_active, axis=1),
            replace=replace
        )

    def _create_session_position(self, df: pd.DataFrame, replace: bool = True) -> pd.DataFrame:
        """
        Cria a feature 'session_position' que representa a posição do log na sessão de forma relativa.
        """
        return self._transform_features(
            df=df,
            required_columns=['num_dia', 'num_log'],
            new_column_name='session_position',
            transform_func=lambda x: (
                x['num_dia'] - 1) + x['num_log'] / x.groupby('num_dia')['num_log'].transform('max'),
            replace=replace
        )

    def _create_engagement_index(self, df: pd.DataFrame, replace: bool = True) -> pd.DataFrame:
        """
        Cria o índice de engajamento como a razão entre 'tempo_equacao' e a soma de 'tempo_equacao' com 'idle_time_acumulado'.
        """
        return self._transform_features(
            df=df,
            required_columns=['tempo_equacao', 'idle_time_acumulado'],
            new_column_name='engagement_index',
            transform_func=lambda x: x['tempo_equacao'] /
            (x['tempo_equacao'] + x['idle_time_acumulado'] + 1e-6),
            replace=replace
        )

    def _create_class_performance_delta(self, df: pd.DataFrame, replace: bool = True) -> pd.DataFrame:
        """
        Cria as features 'delta_tempo_eq' e 'delta_accuracy' para mensurar diferenças de desempenho.
        """
        # Diferença no tempo médio de resolução
        df = self._transform_features(
            df=df,
            required_columns=['tempo_medio_eq_diario',
                              'tempo_medio_eq_diario_turma'],
            new_column_name='delta_tempo_eq',
            transform_func=lambda x: x['tempo_medio_eq_diario'] -
            x['tempo_medio_eq_diario_turma'],
            replace=replace
        )
        # Diferença na acurácia
        df = self._transform_features(
            df=df,
            required_columns=['accuracy_eq',
                              'media_efetividade_passos_eq_turma'],
            new_column_name='delta_accuracy',
            transform_func=lambda x: x['accuracy_eq'] -
            x['media_efetividade_passos_eq_turma'],
            replace=replace
        )
        return df
