import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_predict, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

from model_params import get_models, get_training_configs
from training_constants import CROSS_VALIDATION, GRID_SEARCH, RANDOM_SEARCH, BAYESIAN_OPTIMIZATION
import feature_selection as fs  # Importa o módulo de seleção de características

def train_model(X_train, y_train, training_type, n_iter=50, cv=5, scoring='balanced_accuracy'):
    selectors = fs.create_selectors(X_train, y_train)  # Criar seletores

    models = get_models()
    trained_models = {}  # Dicionário para armazenar os resultados de cada modelo

    training_configs = get_training_configs(n_iter, cv)  # Obter as configurações de treinamento
    config = training_configs[training_type]

    for model_name, model_config in models.items():
        for selector_name, selector in selectors.items():
            
            # Criar pipeline
            pipeline = create_pipeline(selector, model_config)
            
            print(f"\nTraining and evaluating {model_name} with {training_type} and {selector_name}:")

            # Acessar configuração de treinamento
            param_function = config["param_function"]
            if param_function is not None:  # Se houver uma função de parâmetros
                model_specific_args = [param_function()[model_name]]  # Chamar a função de parâmetros e acessar a grade de parâmetros para o modelo atual
            else:
                model_specific_args = []

            # Adicionar parâmetros para o seletor ao espaço de busca
            search_space = fs.get_search_spaces().get(selector_name, {})
            model_specific_args[0].update(search_space)

            if training_type == CROSS_VALIDATION:
                # Tratamento específico para validação cruzada
                best_model, best_result = config["function"](pipeline, X_train, y_train, **config["kwargs"])
            else:
                # Configurar o espaço de busca para todos os outros métodos
                best_model, best_result = config["function"](pipeline, model_specific_args[0], X_train, y_train, **config["kwargs"], scoring=scoring)

            # Armazenando mais informações sobre a configuração
            trained_models[f"{model_name}_{selector_name}"] = {
                'model': best_model,
                'training_type': training_type,
                'hyperparameters': best_model.get_params(),  # Pegando hiperparâmetros do modelo
                'best_result': best_result
            }
            print(f"{training_type} Best Result for {model_name} with {selector_name}: {best_result}")
    
    return trained_models  # Retorna um dicionário com os modelos treinados

def create_pipeline(selector, model_config):
    # Cria o pipeline diretamente com o seletor e o modelo
    pipeline = Pipeline([
        ('feature_selection', selector),
        ('classifier', model_config)
    ])
    return pipeline

def execute_cv(model, X_train, y_train, cv=5):
    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv)
    model.fit(X_train, y_train)
    return model, balanced_accuracy_score(y_train, y_pred_cv), y_pred_cv

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

# Função de otimização bayesiana com BayesSearchCV
def execute_bayesian_optimization(model, space, X_train, y_train, n_iter=50, cv=5, scoring='balanced_accuracy'):
    search = BayesSearchCV(
        model,
        search_spaces=space,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,  # Ajustar aqui a métrica de avaliação
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    return best_model, search.best_score_

