from abc import ABC, abstractmethod

class BaseFeatureSelector(ABC):
    def __init__(self, X_train, y_train=None, **kwargs):
        self.X_train = X_train
        self.y_train = y_train
        self.selector = self._create_selector(**kwargs)
    
    @abstractmethod
    def _create_selector(self, **kwargs):
        """
        Método abstrato para criar o seletor de características.
        Deve ser implementado por todas as subclasses.
        """
        pass

    @abstractmethod
    def get_search_space(self):
        """
        Método abstrato para retornar o espaço de busca de hiperparâmetros.
        Deve ser implementado por todas as subclasses.
        """
        pass
