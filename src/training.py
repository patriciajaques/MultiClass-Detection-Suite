import numpy as np
from sklearn import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from model_params import get_models, get_param_grids, get_hyperopt_spaces, get_param_distributions
from training_constants import CROSS_VALIDATION, GRID_SEARCH, RANDOM_SEARCH, BAYESIAN_OPTIMIZATION


def train_model(X_train, y_train, training_type, pipeline):
    models = get_models()

    # Configurações de treinamento unificadas
    training_configs = {
        CROSS_VALIDATION: {
            "function": execute_cv,
            "param_function": None,  # Não há função de parâmetros para cross validation
            "kwargs": {}
        },
        GRID_SEARCH: {
            "function": execute_grid_search,
            "param_function": get_param_grids,  # Função que retorna os parâmetros para grid search
            "kwargs": {"cv": 5}
        },
        RANDOM_SEARCH: {
            "function": execute_random_search,
            "param_function": get_param_distributions,  # Função que retorna os parâmetros para random search
            "kwargs": {"n_iter": 50, "cv": 5}
        },
        BAYESIAN_OPTIMIZATION: {
            "function": execute_bayesian_optimization,
            "param_function": get_hyperopt_spaces,  # Função que retorna os parâmetros para bayesian optimization
            "kwargs": {"max_evals": 50}
        }
    }

    trained_models = {}  # Dicionário para armazenar os resultados de cada modelo

    config = training_configs[training_type]
    

    for model_name, model_config in models.items():
        # Clona o pipeline para que ele não seja modificado, garantindo isolamento entre os modelos
        # Cada modelo terá seu próprio pipeline independente, garantindo que as alterações nos hiperparâmetros durante a otimização de um modelo não afetem os outros.
        pipeline = clone(pipeline) 

        # Substitui o classificador no pipeline
        pipeline.steps[-1] = ('classifier', model_config)
        print(f"\nTraining and evaluating {model_name} with {training_type}:")
        
        # Acessar configuração de treinamento
        param_function = config["param_function"]
        if param_function is not None:  # Se houver uma função de parâmetros
            model_specific_args = [param_function()[model_name]]  # Chamar a função de parâmetros e acessar a grade de parâmetros para o modelo atual
        else:
            model_specific_args = []
        best_model, best_result = config["function"](pipeline, *model_specific_args, X_train, y_train, **config["kwargs"])

        # Armazenando mais informações sobre a configuração
        trained_models[model_name] = {
            'model': best_model,
            'training_type': training_type,
            'hyperparameters': pipeline.get_params(),  # Pegando hiperparâmetros do modelo
            'best_result': best_result
        }
        print(f"{training_type} Best Result for {model_name}: {best_result}")
    
    return trained_models  # Retorna um dicionário com os modelos treinados

def execute_cv(model, X_train, y_train, cv=5):
    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv)
    model.fit(X_train, y_train)
    return model, balanced_accuracy_score(y_train, y_pred_cv)

def execute_grid_search(model, param_grid, X_train, y_train, cv=5):
    print(f"Grid search for {model['classifier']}")
    print(f"Param grid: {param_grid}")
    search = GridSearchCV(model, param_grid, cv=cv, scoring='balanced_accuracy', verbose=1, n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    return best_model, search.best_score_


def execute_random_search(model, param_distributions, X_train, y_train, n_iter=50, cv=5):
    search = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, cv=cv, scoring='balanced_accuracy', verbose=1, n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    return best_model, search.best_score_

# Função de otimização bayesiana
def execute_bayesian_optimization(model, space, X_train, y_train, max_evals=50):
    # Função utilitária para conversão de parâmetros
    def convert_hyperopt_params(params, space):
        """Converte índices para strings em parâmetros selecionados pelo Hyperopt."""
        for key, val in params.items():
            if isinstance(val, int) and key in space and isinstance(space[key], hp.choice):
                params[key] = space[key].pos_args[val]
        return params

    def objective(params):
        """Função objetivo para otimização bayesiana."""
        params = convert_hyperopt_params(params, space)
        model.set_params(**params)
        score = cross_val_score(model, X_train, y_train, scoring='balanced_accuracy', cv=5)
        return {'loss': -np.mean(score), 'status': STATUS_OK}
    
    trials = Trials()
    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    best_params = convert_hyperopt_params(best_params, space)
    model.set_params(**best_params)
    model.fit(X_train, y_train)  # Treine o modelo com os melhores parâmetros
    return model, -trials.best_trial['result']['loss']
