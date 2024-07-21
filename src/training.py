
import numpy as np
from sklearn.metrics import confusion_matrix
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, cross_validate
import logging

from model_params import get_models, get_bayes_search_spaces
import feature_selection as fs  # Importa o módulo de seleção de características
from logger_config import LoggerConfig

# Configuração do logger
LoggerConfig.configure_log_file('bayesian_optimization', '.log')

# constants
CROSS_VALIDATION = 'Cross-Validation'
BAYESIAN_OPTIMIZATION = 'Bayesian Optimization'

def train_model(X_train, y_train, training_type, n_iter=50, cv=5, scoring='balanced_accuracy'):
    selectors = fs.create_selectors(X_train, y_train)  # Criar seletores

    models = get_models()
    trained_models = {}  # Dicionário para armazenar os resultados de cada modelo

    for model_name, model_config in models.items():
        for selector_name, selector in selectors.items():
            
            # Criar pipeline
            pipeline = create_pipeline(selector, model_config)
            
            logging.info(f"Training and evaluating {model_name} with {training_type} and {selector_name}:")
            print(f"Training and evaluating {model_name} with {training_type} and {selector_name}:")

            if training_type == CROSS_VALIDATION:
                # Tratamento específico para validação cruzada
                best_model, best_result, y_pred_cv, cv_results = execute_cv(model_config, X_train, y_train, cv=cv)
            else: # Bayesian Optimization
                search_space = get_bayes_search_spaces()[model_name]
                logging.info(f"Running Bayesian optimization for {model_name} with selector {selector_name}")
                logging.info(f"Search space: {search_space}")
                # Adicionar parâmetros para o seletor ao espaço de busca
                search_space.update(fs.get_search_spaces().get(selector_name, {}))
                # Best model será um objeto do tipo Pipeline
                best_model, best_result, opt = execute_bayesian_optimization(pipeline, search_space, X_train, y_train, n_iter=n_iter, cv=cv, scoring=scoring)
                # Extraindo as predições de validação cruzada diretamente de cv_results_
                logging.info(f"Bayesian optimization results: Melhor resultado: {best_result}, Resultado médio da validação cruzada: {cv_results['mean_test_score'][opt.best_index_]}")

            # Armazenando mais informações sobre a configuração
            trained_models[f"{model_name}_{selector_name}"] = {
                'model': best_model,
                'training_type': training_type,
                'hyperparameters': best_model.get_params(),  # Pegando hiperparâmetros do modelo
                'cv_result': best_result
            }
            logging.info(f"{training_type} Best Result for {model_name} with {selector_name}: {best_result}")
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
    # Usando cross_validate para obter as pontuações de cada fold
    cv_results = cross_validate(model, X_train, y_train, cv=cv, return_estimator=True, scoring='balanced_accuracy')
    
    # Ajustando o modelo ao conjunto de treinamento completo
    model.fit(X_train, y_train)
    
    # Calculando as predições usando cross_val_predict para consistência com o código original
    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv)
    
    # Retornando o modelo treinado, a média das pontuações de acurácia balanceada, as predições e os resultados da validação cruzada
    return model, np.mean(cv_results['test_score']), y_pred_cv, cv_results

# Função de otimização bayesiana com BayesSearchCV
def execute_bayesian_optimization(model, space, X_train, y_train, n_iter=50, cv=5, scoring='balanced_accuracy'):
    search = BayesSearchCV(
        model,
        search_spaces=space,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=2,
        random_state=42,
        verbose=3,
        return_train_score=True
    )

    search.fit(X_train, y_train, callback=log_results)
    best_model = search.best_estimator_

    return best_model, search.best_score_, search

def log_results(result):
    """
    Registra os parâmetros testados e a pontuação para cada iteração.
    Inverte a pontuação se ela for negativa, apenas para exibição.
    """
    if len(result.x_iters) > 0:  # Verificar se há iterações para logar
        # Inverter o sinal da pontuação para exibição se ela for negativa
        score = -result.func_vals[-1] if result.func_vals[-1] < 0 else result.func_vals[-1]
        logging.info(f"Iteration {len(result.x_iters)}: tested parameters: {result.x_iters[-1]}, score: {score}")
