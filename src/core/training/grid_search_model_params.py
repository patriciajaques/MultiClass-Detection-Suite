from core.training.model_params import ModelParams


class GridSearchModelParams():
    @staticmethod
    def get_param_grid(model_name):
        param_space = ModelParams.get_param_space(model_name)
        param_grid = {}
        for param, space in param_space.items():
            if isinstance(space, tuple):
                if isinstance(space[0], int):
                    param_grid[param] = list(range(space[0], space[1]+1))
                else:
                    param_grid[param] = [space[0], (space[0]+space[1])/2, space[1]]
            elif isinstance(space, list):
                param_grid[param] = space
        return param_grid