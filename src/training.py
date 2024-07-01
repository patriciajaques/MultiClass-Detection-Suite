import logging
from datetime import datetime
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_predict

# Gerar um nome de arquivo com data e hora
log_filename = datetime.now().strftime('bayesian_optimization_%Y%m%d_%H%M.log')

# Configuração do logging
logging.basicConfig(
    filename=log_filename,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

from model_params import get_models, get_bayes_search_spaces
import feature_selection as fs  # Importa o módulo de seleção de características

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
                best_model, best_result = execute_cv(model_config, X_train, y_train, cv=cv)
            else: # Bayesian Optimization
                search_space = get_bayes_search_spaces()[model_name]
                logging.info(f"Running Bayesian optimization for {model_name} with selector {selector_name}")
                logging.info(f"Search space: {search_space}")
                # Adicionar parâmetros para o seletor ao espaço de busca
                search_space.update(fs.get_search_spaces().get(selector_name, {}))

                best_model, best_result = execute_bayesian_optimization(pipeline, search_space, X_train, y_train, n_iter=n_iter, cv=cv, scoring=scoring)
                logging.info(f"Bayesian optimization results: {best_result}")

            # Armazenando mais informações sobre a configuração
            trained_models[f"{model_name}_{selector_name}"] = {
                'model': best_model,
                'training_type': training_type,
                'hyperparameters': best_model.get_params(),  # Pegando hiperparâmetros do modelo
                'best_result': best_result
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
    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv)
    model.fit(X_train, y_train)
    return model, balanced_accuracy_score(y_train, y_pred_cv), y_pred_cv

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
        verbose=3
    )

    search.fit(X_train, y_train, callback=log_results)
    best_model = search.best_estimator_
    return best_model, search.best_score_

def log_results(result):
    """
    Registra os parâmetros testados e a pontuação para cada iteração.
    Inverte a pontuação se ela for negativa, apenas para exibição.
    """
    if len(result.x_iters) > 0:  # Verificar se há iterações para logar
        # Inverter o sinal da pontuação para exibição se ela for negativa
        score = -result.func_vals[-1] if result.func_vals[-1] < 0 else result.func_vals[-1]
        logging.info(f"Iteration {len(result.x_iters)}: tested parameters: {result.x_iters[-1]}, score: {score}")
