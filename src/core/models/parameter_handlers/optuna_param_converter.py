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

    # src/core/models/parameter_handlers/optuna_param_converter.py


    def suggest_selector_hyperparameters(trial, selector_search_space):
        """Suggests hyperparameters for feature selector with improved handling."""
        if not selector_search_space:
            return {}

        suggested_params = {}

        for param_name, param_config in selector_search_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type')
                if param_type == 'float':
                    if 'values' in param_config:
                        suggested_params[param_name] = trial.suggest_categorical(
                            param_name, param_config['values']
                        )
                    else:
                        range_vals = param_config['range']
                        suggested_params[param_name] = trial.suggest_float(
                            param_name, range_vals[0], range_vals[1]
                        )
                elif param_type == 'int':
                    range_vals = param_config['range']
                    suggested_params[param_name] = trial.suggest_int(
                        param_name, range_vals[0], range_vals[1]
                    )
            elif isinstance(param_config, list):
                suggested_params[param_name] = trial.suggest_categorical(
                    param_name, param_config
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