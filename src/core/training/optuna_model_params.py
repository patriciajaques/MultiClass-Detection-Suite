from core.training.model_params import ModelParams

class OptunaModelParams():
    
    @staticmethod
    def suggest_hyperparameters(trial, model_name):
        param_space = ModelParams.get_param_space(model_name)
        suggested_params = {}
        for param, space in param_space.items():
            if isinstance(space, tuple):
                if isinstance(space[0], int):
                    suggested_params[param] = trial.suggest_int(param, space[0], space[1])
                else:
                    suggested_params[param] = trial.suggest_float(param, space[0], space[1], log=space[2]=='log-uniform')
            elif isinstance(space, list):
                suggested_params[param] = trial.suggest_categorical(param, space)
        return suggested_params
