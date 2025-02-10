import numpy as np


class GridSearchParamConverter:
    @staticmethod
    def convert_param_space(model_params, model_name):
        param_space = model_params.get_param_space(model_name)

        # Se for uma lista de dicionários (múltiplos espaços de parâmetros)
        if isinstance(param_space, list):
            # Combina todos os dicionários em um único espaço
            combined_space = {}
            for space in param_space:
                combined_space.update(space)
            param_space = combined_space

        # Converte cada parâmetro individualmente
        return {k: GridSearchParamConverter._convert_single_param(v)
                for k, v in param_space.items()}

    @staticmethod
    def _convert_single_param(space):
        """Converte um único espaço de parâmetro para o formato do GridSearchCV."""
        # Se for um dicionário com configuração específica
        if isinstance(space, dict):
            if 'values' in space:
                return space['values']
            elif 'range' in space:
                start, end = space['range']
                if space.get('type') == 'int':
                    return list(range(start, end + 1))
                elif space.get('type') == 'float':
                    return np.linspace(start, end, 5).tolist()
            return [space]  # Para outros tipos de dicionários

        # Se for uma tupla (intervalo)
        elif isinstance(space, tuple):
            start, end = space[:2]
            if isinstance(start, int):
                return list(range(start, end + 1))
            return np.linspace(start, end, 5).tolist()

        # Se já for uma lista
        elif isinstance(space, list):
            return space

        # Para valores únicos
        return [space]

    @staticmethod
    def combine_with_selector_space(param_grid, selector_space):
        """Combina o espaço do modelo com o espaço do seletor."""
        if not selector_space:
            return param_grid

        selector_grid = {k: GridSearchParamConverter._convert_single_param(v)
                         for k, v in selector_space.items()}

        param_grid.update(selector_grid)
        return param_grid
