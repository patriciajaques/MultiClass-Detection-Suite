"""
Copyright (c) 2025 Patricia Jaques
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License. To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
"""

from core.models.multiclass.multiclass_model_params import MulticlassModelParams

class DigitsModelParams(MulticlassModelParams):
    """
    Classe para parâmetros de modelos específicos para o dataset MNIST.
    
    Esta classe herda todas as configurações da classe MulticlassModelParams sem modificações.
    Atualmente utiliza os mesmos parâmetros da classe pai para todos os modelos.
    
    Note:
        Se futuramente for necessário customizar parâmetros específicos para o MNIST,
        basta sobrescrever os métodos relevantes nesta classe.
    """
    def get_models(self):
        """
        Retorna o dicionário de modelos base.
        Utiliza a implementação da classe pai MulticlassModelParams.
        """
        return super().get_models()

    def get_param_space(self, model_name):
        """
        Retorna o espaço de parâmetros para um modelo específico.
        Utiliza a implementação da classe pai MulticlassModelParams.
        """
        return super().get_param_space(model_name)