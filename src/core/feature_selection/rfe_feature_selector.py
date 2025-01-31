from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.base import clone
from core.feature_selection.base_feature_selector import BaseFeatureSelector


class RFEFeatureSelector(BaseFeatureSelector):
    """
    Implementa seleção de features usando RFE (Recursive Feature Elimination)
    com validação cruzada para encontrar o número ótimo de features.
    """

    def __init__(self, X_train=None, y_train=None, min_features_to_select=10, step=0.1):
        """
        Inicializa o RFE Feature Selector.
        
        Args:
            X_train: Dados de treino
            y_train: Labels de treino
            min_features_to_select: Número mínimo de features para selecionar
            step: Fração de features a remover a cada iteração (entre 0 e 1)
        """
        self.min_features_to_select = min_features_to_select
        self.step = step
        self._is_fitted = False
        super().__init__(X_train=X_train, y_train=y_train)

    def _create_selector(self):
        """
        Cria e configura o seletor RFE com validação cruzada.
        """
        # Configura o estimador base com parâmetros conservadores
        base_estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=1,  # Importante: use 1 aqui para evitar conflitos com paralelização externa
            random_state=42
        )

        # Se X_train estiver disponível, ajusta min_features_to_select
        if self.X_train is not None:
            n_features = self.X_train.shape[1]
            if isinstance(self.min_features_to_select, float):
                self.min_features_to_select = max(
                    1, int(self.min_features_to_select * n_features))

        # Cria o seletor RFECV
        selector = RFECV(
            estimator=base_estimator,
            step=self.step,
            min_features_to_select=self.min_features_to_select,
            cv=5,  # 5-fold CV por padrão
            scoring='balanced_accuracy',
            n_jobs=1  # Importante: use 1 aqui para evitar conflitos
        )

        return selector

    def fit(self, X, y):
        """
        Ajusta o seletor RFE aos dados.
        
        Args:
            X: Features para treino
            y: Labels para treino
            
        Returns:
            self: Retorna a instância ajustada
        """
        self.X_train = X
        self.y_train = y

        if self.selector is None:
            self.selector = self._create_selector()

        # Ajusta o seletor e registra o número de features selecionadas
        self.selector.fit(X, y)
        self._is_fitted = True
        self.n_features_selected_ = self.selector.n_features_

        return self

    def transform(self, X):
        """
        Transforma os dados usando apenas as features selecionadas.
        
        Args:
            X: Dados para transformar
            
        Returns:
            array: Dados transformados apenas com as features selecionadas
        """
        if not self._is_fitted:
            raise ValueError("Seletor não foi ajustado. Execute fit primeiro.")
        return self.selector.transform(X)

    def get_support(self):
        """
        Retorna máscara booleana das features selecionadas.
        """
        if not self._is_fitted:
            raise ValueError("Seletor não foi ajustado. Execute fit primeiro.")
        return self.selector.support_

    def get_feature_names(self, feature_names=None):
        """
        Retorna os nomes das features selecionadas.
        
        Args:
            feature_names: Lista opcional com nomes das features originais
            
        Returns:
            list: Nomes das features selecionadas
        """
        mask = self.get_support()
        if feature_names is not None:
            return [f for f, s in zip(feature_names, mask) if s]
        return [i for i, s in enumerate(mask) if s]

    def get_search_space(self):
        """
        Define o espaço de busca para otimização de hiperparâmetros.
        """
        search_space = {
            'feature_selection__min_features_to_select': {
                'type': 'int',
                'range': [5, 50]  # Ajuste conforme necessário
            },
            'feature_selection__step': {
                'type': 'float',
                'values': [0.1, 0.2, 0.3]  # Valores discretos para step
            }
        }
        return search_space

    def set_params(self, **params):
        """
        Atualiza os parâmetros do seletor.
        """
        if 'min_features_to_select' in params:
            self.min_features_to_select = params['min_features_to_select']
        if 'step' in params:
            self.step = params['step']

        # Se já estiver ajustado, recria o seletor com os novos parâmetros
        if self._is_fitted:
            self.selector = self._create_selector()
            self.selector.fit(self.X_train, self.y_train)

        return self

    def get_params(self, deep=True):
        """
        Retorna os parâmetros atuais do seletor.
        """
        return {
            'min_features_to_select': self.min_features_to_select,
            'step': self.step
        }
