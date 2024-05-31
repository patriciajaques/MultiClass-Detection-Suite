import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from model_params import get_models, get_param_grids, get_hyperopt_spaces, get_param_distributions
from preprocessing import create_preprocessor
from training_constants import CROSS_VALIDATION, GRID_SEARCH, RANDOM_SEARCH, BAYESIAN_OPTIMIZATION

def objective(params, clf, X_train, y_train):
    """Função objetivo para otimização bayesiana."""
    model = clf.set_params(**params)
    score = cross_val_score(model, X_train, y_train, scoring='balanced_accuracy', cv=5)
    return {'loss': -np.mean(score), 'status': STATUS_OK}

def train_model(X_train, y_train, training_type):
    models = get_models()
    preprocessor = create_preprocessor(X_train)

    # Configurações de treinamento unificadas
    training_configs = {
        CROSS_VALIDATION: {
            "function": execute_cv,
            "args": [X_train, y_train],
            "kwargs": {}
        },
        GRID_SEARCH: {
            "function": execute_grid_search,
            "args": [get_param_grids(), X_train, y_train],
            "kwargs": {"cv": 5}
        },
        RANDOM_SEARCH: {
            "function": execute_random_search,
            "args": [get_param_distributions(), X_train, y_train],
            "kwargs": {"n_iter": 50, "cv": 5}
        },
        BAYESIAN_OPTIMIZATION: {
            "function": execute_bayesian_optimization,
            "args": [lambda params: objective(params, model, X_train, y_train), get_hyperopt_spaces(), X_train, y_train],
            "kwargs": {"max_evals": 50}
        }
    }

    results = {}  # Dicionário para armazenar os resultados de cada modelo

    for model_name, model_config in models.items():
        model = Pipeline([('preprocessor', preprocessor), ('classifier', model_config)])
        print(f"\nTraining and evaluating {model_name} with {training_type}:")
        
        # Acessar configuração de treinamento
        config = training_configs[training_type]
        best_model, best_result = config["function"](model, *config["args"], **config["kwargs"])

        # Armazenar o melhor modelo e resultado no dicionário
        results[model_name] = {
            "model": best_model,
            "result": best_result
        }
        print(f"{training_type} Best Result for {model_name}: {best_result}")
    
    return results  # Retorna o dicionário com todos os resultados


def execute_cv(model, X_train, y_train, cv=5):
    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv)
    model.fit(X_train, y_train)
    return model, balanced_accuracy_score(y_train, y_pred_cv)

def execute_grid_search(model, param_grid, X_train, y_train, cv=5):
    search = GridSearchCV(model, param_grid, cv=cv, scoring='balanced_accuracy', verbose=1, n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    return best_model, search.best_score_

def execute_random_search(model, param_distributions, X_train, y_train, n_iter=50, cv=5):
    search = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, cv=cv, scoring='balanced_accuracy', verbose=1, n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    return best_model, search.best_score_

def execute_bayesian_optimization(model, objective, space, X_train, y_train, max_evals=50):
    trials = Trials()
    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    # Convert the best parameters to a format that can be used with the model
    best_params = {k: v[0] if isinstance(v, list) else v for k, v in best_params.items()}
    model.set_params(**best_params)
    model.fit(X_train, y_train) # Train the model with the best parameters
    return model, trials.best_trial['result']['loss']