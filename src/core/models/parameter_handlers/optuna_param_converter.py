import logging

class OptunaParamConverter:
    # Usar o mesmo logger do OptunaBayesianOptimizationTraining
    logger = logging.getLogger('optuna_training')

    @staticmethod
    def suggest_parameters(trial, model_params, model_name):
        """
        Sugere parâmetros para o Optuna com base no espaço de parâmetros do modelo.
        
        Args:
            trial: Trial do Optuna
            model_params (BaseModelParams): Instância que define os parâmetros do modelo
            model_name (str): Nome do modelo
            
        Returns:
            dict: Dicionário com os parâmetros sugeridos
        """
        param_space = model_params.get_param_space(model_name)
        
        # Se for uma lista de dicionários (como no caso da Regressão Logística)
        if isinstance(param_space, list):
            # Combina todos os dicionários em um único espaço de busca
            combined_space = {}
            for param_dict in param_space:
                combined_space.update(param_dict)
            param_space = combined_space
            
        return OptunaParamConverter._suggest_from_space(trial, param_space)

    def suggest_selector_hyperparameters(trial, selector_search_space):
        """
        Sugere hiperparâmetros para o seletor de features baseado no espaço de busca fornecido.
        """
        if not selector_search_space:
            return {}

        suggested_params = {}
        
        for param_name, param_values in selector_search_space.items():
            if isinstance(param_values, list):
                # Se todos os valores são float, usar suggest_float
                if all(isinstance(x, float) for x in param_values):
                    suggested_params[param_name] = trial.suggest_float(
                        param_name,
                        min(param_values),
                        max(param_values)
                    )
                # Se todos os valores são int, usar suggest_int
                elif all(isinstance(x, int) for x in param_values):
                    suggested_params[param_name] = trial.suggest_int(
                        param_name,
                        min(param_values),
                        max(param_values)
                    )
                # Caso contrário, usar categorical
                else:
                    suggested_params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_values
                    )
            else:
                # Para outros tipos de parâmetros
                suggested_params[param_name] = OptunaParamConverter._suggest_single_parameter(
                    trial, param_name, param_values
                )

        return suggested_params

    @staticmethod
    def _suggest_from_space(trial, param_space):
        """
        Método auxiliar para sugerir parâmetros baseado no espaço de busca.
        
        Args:
            trial: Trial do Optuna
            param_space (dict): Dicionário com o espaço de busca dos parâmetros
            
        Returns:
            dict: Dicionário com os parâmetros sugeridos
        """
        if not param_space:
            return {}
            
        suggested_params = {}
        
        for param_name, param_values in param_space.items():
            try:
                suggested_params[param_name] = OptunaParamConverter._suggest_single_parameter(
                    trial, param_name, param_values
                )
            except Exception as e:
                # Usar o logger da classe ao invés de logging diretamente
                OptunaParamConverter.logger.warning(f"Erro ao sugerir parâmetro {param_name}: {str(e)}")
                
                
        return suggested_params

    @staticmethod
    def _suggest_single_parameter(trial, param_name, param_values):
        """
        Sugere um único parâmetro baseado em seus valores possíveis.
        
        Args:
            trial: Trial do Optuna
            param_name (str): Nome do parâmetro
            param_values (list|tuple): Valores possíveis para o parâmetro
            
        Returns:
            Valor sugerido para o parâmetro
        """
        if isinstance(param_values, list):
            return trial.suggest_categorical(param_name, param_values)
        
        elif isinstance(param_values, tuple):
            if isinstance(param_values[0], int):
                return trial.suggest_int(param_name, param_values[0], param_values[1])
            else:
                is_log = len(param_values) > 2 and param_values[2] == 'log-uniform'
                return trial.suggest_float(param_name, param_values[0], param_values[1], log=is_log)
        
        raise ValueError(f"Tipo de valor não suportado para o parâmetro {param_name}: {type(param_values)}")