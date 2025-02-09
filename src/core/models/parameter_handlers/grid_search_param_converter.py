class GridSearchParamConverter:
    @staticmethod
    def convert_param_space(model_params, model_name):
        param_space = model_params.get_param_space(model_name)

        # Se for uma lista, percorra e converta cada espaço de parâmetro individualmente
        if isinstance(param_space, list):
            return [
                {k: GridSearchParamConverter._convert_single_param(
                    v) for k, v in space.items()}
                for space in param_space
            ]
        else:
            # Se for um dicionário, converta diretamente
            return {k: GridSearchParamConverter._convert_single_param(v) for k, v in param_space.items()}

    @staticmethod
    def _convert_single_param(space):
        """Converte um único espaço de parâmetro para o formato esperado pelo GridSearchCV."""
        if isinstance(space, dict):
            if 'values' in space:
                return space['values']
            elif space.get('type') == 'int' and 'range' in space:
                start, end = space['range']
                return list(range(start, end + 1))
            elif space.get('type') == 'float' and 'values' in space:
                return space['values']
            elif space.get('type') == 'float' and 'range' in space:
                # Aqui você pode definir como gerar valores para floats sem 'values'
                # Por exemplo, gerar 3 valores igualmente espaçados:
                start, end = space['range']
                return [start, (start + end) / 2, end]
            else:
                # Se não corresponder a nenhum padrão conhecido, retorna o dicionário encapsulado em lista
                return [space]
        elif isinstance(space, tuple):
            if isinstance(space[0], int):
                return list(range(space[0], space[1] + 1))
            return [space[0], (space[0] + space[1]) / 2, space[1]]
        elif isinstance(space, list):
            return space
        return [space]


    @staticmethod
    def convert_selector_param_space(selector_space):
        """
        Converte o espaço de busca dos hiperparâmetros do seletor para o formato esperado
        pelo GridSearchCV.
        """
        # Se for um dicionário, converte cada valor individualmente
        if isinstance(selector_space, dict):
            return {k: GridSearchParamConverter._convert_single_param(v) for k, v in selector_space.items()}
        # Se for uma lista de espaços, converte cada um
        elif isinstance(selector_space, list):
            return [GridSearchParamConverter.convert_selector_param_space(item) for item in selector_space]
        else:
            return selector_space