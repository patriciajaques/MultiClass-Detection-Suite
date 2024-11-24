
class GridSearchParamConverter:
    @staticmethod
    def convert_param_space(model_params, model_name):
        param_space = model_params.get_param_space(model_name)
        if isinstance(param_space, list):
            # Se param_space for uma lista, retorne-a diretamente
            return param_space
        if isinstance(param_space, list):
            return [
                {k: GridSearchParamConverter._convert_single_param(v) for k, v in space.items()}
                for space in param_space
            ]
        else:
            return {k: GridSearchParamConverter._convert_single_param(v) for k, v in param_space.items()}

    @staticmethod
    def _convert_single_param(space):
        """Converte um único espaço de parâmetro para o formato do GridSearchCV."""
        if isinstance(space, tuple):
            if isinstance(space[0], int):
                return list(range(space[0], space[1] + 1))
            return [space[0], (space[0] + space[1]) / 2, space[1]]
        elif isinstance(space, list):
            return space
        return [space]