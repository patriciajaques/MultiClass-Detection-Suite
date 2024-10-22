from core.training.model_params import ModelParams

class OptunaModelParams:
    @staticmethod
    def suggest_model_hyperparameters(trial, model_name):
        param_space = ModelParams.get_param_space(model_name)
        return OptunaModelParams._suggest_params_from_space(trial, param_space)

    @staticmethod
    def suggest_selector_hyperparameters(trial, selector_search_space):
        return OptunaModelParams._suggest_params_from_space(trial, selector_search_space)

    @staticmethod
    def _suggest_params_from_space(trial, param_space):
        suggested_params = {}
        
        if isinstance(param_space, list):
            param_space = trial.suggest_categorical('param_space', param_space)
            
        for param, values in param_space.items():
            if isinstance(values, tuple):
                if isinstance(values[0], int):
                    suggested_params[param] = trial.suggest_int(param, values[0], values[1])
                else:
                    suggested_params[param] = trial.suggest_float(
                        param, values[0], values[1], 
                        log=values[2]=='log-uniform' if len(values) > 2 else False
                    )
            elif isinstance(values, list):
                if values and isinstance(values[0], int):
                    suggested_params[param] = trial.suggest_int(param, min(values), max(values))
                elif values and isinstance(values[0], float):
                    suggested_params[param] = trial.suggest_float(param, min(values), max(values))
                else:
                    suggested_params[param] = trial.suggest_categorical(param, values)
                    
        return suggested_params