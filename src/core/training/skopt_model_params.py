from core.training.model_params import ModelParams

class SkoptModelParams():

    @classmethod
    def get_bayes_search_spaces(cls):
        return {model: ModelParams.get_param_space(model) for model in ModelParams.get_models()}