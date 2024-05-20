from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import make_scorer, balanced_accuracy_score

import utils
import data_exploration as de
import preprocessing as pp

def load_data (file_path = '../data/new_logs_labels.csv'):
    """
    Lê um arquivo CSV com delimitador ';' e inspeciona seu conteúdo.

    Args:
        file_path (str): Caminho para o arquivo CSV.
    
    Returns:
        pd.DataFrame: DataFrame contendo os dados lidos.
    """
    
    df = de.load_data('../data/new_logs_labels.csv')
    X, y = utils.split_features_and_target(df)
    y = y['comportamento']
    X.info()
    y.info()

    return X, y

def split_train_test_data (X, y):

    """
    Divide os dados em conjuntos de treino e teste.

    Args:
        X (pd.DataFrame): DataFrame contendo as features.
        y (pd.DataFrame): DataFrame contendo o target.
    
    Returns:
        X_train (pd.DataFrame): DataFrame contendo as features de treino.
        X_test (pd.DataFrame): DataFrame contendo as features de teste.
        y_train (pd.DataFrame): DataFrame contendo o target de treino.
        y_test (pd.DataFrame): DataFrame contendo o target de teste.
    """
    
    # Cria um novo dataframe que contém y concatenado com X
    data = de.concat_features_and_target(X, y)
    train_data, test_data = utils.split_data_stratified(data, test_size=0.3, target_column='aluno', n_splits=5)
    # Separar features e rótulos
    X_train = train_data.drop(columns=['comportamento'])
    y_train = train_data['comportamento']
    X_test = test_data.drop(columns=['comportamento'])
    y_test = test_data['comportamento']
    return X_train, X_test, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test, enable_cv=True, enable_gridsearch=True, enable_randomsearch=True, enable_bayesian_optimization=True):
    """
    Recebe a lista de modelos e chama o método execute_pipeline para cada modelo.

    Args:
    X_train (DataFrame): Conjunto de treinamento.
    X_test (DataFrame): Conjunto de teste.
    y_train (Series): Rótulos de treinamento.
    y_test (Series): Rótulos de teste.
    enable_cv (bool): Habilitar/Desabilitar validação cruzada.
    enable_gridsearch (bool): Habilitar/Desabilitar GridSearchCV.
    enable_randomsearch (bool): Habilitar/Desabilitar RandomizedSearchCV.
    enable_bayesian_optimization (bool): Habilitar/Desabilitar Bayesian Optimization.

    Returns:
    None
    """

    # Codificar rótulos
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Identificar colunas numéricas e categóricas
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

    # Criar o pré-processador
    preprocessor = pp.create_preprocessor(numeric_features, categorical_features)

    # Lista de modelos para treinamento e avaliação
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

    # Definir as grades e distribuições de parâmetros
    param_grids = {
        'Logistic Regression': {
            'classifier__penalty': ['l1', 'l2', 'elasticnet', 'None'],
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__solver': ['lbfgs', 'liblinear', 'saga']
        },
        'Decision Tree': {
            'classifier__max_depth': [None, 10, 20, 30, 40, 50],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        },
        'Random Forest': {
            'classifier__n_estimators': [10, 50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'classifier__n_estimators': [10, 50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 10],
            'classifier__subsample': [0.5, 0.7, 1.0]
        },
        'SVM': {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'classifier__gamma': ['scale', 'auto']
        },
        'KNN': {
            'classifier__n_neighbors': [3, 5, 10, 20],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'XGBoost': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7],
            'classifier__subsample': [0.7, 0.8, 1.0],
            'classifier__colsample_bytree': [0.7, 0.8, 1.0]
        }
    }

    param_distributions = {
        'Logistic Regression': {
            'classifier__penalty': ['l1', 'l2', 'elasticnet', 'None'],
            'classifier__C': randint(0.1, 100),
            'classifier__solver': ['lbfgs', 'liblinear', 'saga']
        },
        'Decision Tree': {
            'classifier__max_depth': randint(1, 50),
            'classifier__min_samples_split': randint(2, 10),
            'classifier__min_samples_leaf': randint(1, 4)
        },
        'Random Forest': {
            'classifier__n_estimators': randint(10, 200),
            'classifier__max_depth': randint(1, 30),
            'classifier__min_samples_split': randint(2, 10),
            'classifier__min_samples_leaf': randint(1, 4)
        },
        'Gradient Boosting': {
            'classifier__n_estimators': randint(10, 200),
            'classifier__learning_rate': hp.uniform('classifier__learning_rate', 0.01, 0.2),
            'classifier__max_depth': randint(3, 10),
            'classifier__subsample': hp.uniform('classifier__subsample', 0.5, 1.0)
        },
        'SVM': {
            'classifier__C': randint(0.1, 100),
            'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'classifier__gamma': ['scale', 'auto']
        },
        'KNN': {
            'classifier__n_neighbors': randint(3, 20),
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'XGBoost': {
            'classifier__n_estimators': randint(100, 300),
            'classifier__learning_rate': hp.uniform('classifier__learning_rate', 0.01, 0.2),
            'classifier__max_depth': randint(3, 7),
            'classifier__subsample': hp.uniform('classifier__subsample', 0.7, 1.0),
            'classifier__colsample_bytree': hp.uniform('classifier__colsample_bytree', 0.7, 1.0)
        }
    }

    space = {
        'Logistic Regression': {
            'penalty': hp.choice('penalty', ['l1', 'l2', 'elasticnet', 'None']),
            'C': hp.loguniform('C', np.log(0.1), np.log(100)),
            'solver': hp.choice('solver', ['lbfgs', 'liblinear', 'saga'])
        },
        'Decision Tree': {
            'max_depth': hp.randint('max_depth', 1, 50),
            'min_samples_split': hp.randint('min_samples_split', 2, 10),
            'min_samples_leaf': hp.randint('min_samples_leaf', 1, 4)
        },
        'Random Forest': {
            'n_estimators': hp.randint('n_estimators', 10, 200),
            'max_depth': hp.randint('max_depth', 1, 30),
            'min_samples_split': hp.randint('min_samples_split', 2, 10),
            'min_samples_leaf': hp.randint('min_samples_leaf', 1, 4)
        },
        'Gradient Boosting': {
            'n_estimators': hp.randint('n_estimators', 10, 200),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'max_depth': hp.randint('max_depth', 3, 10),
            'subsample': hp.uniform('subsample', 0.5, 1.0)
        },
        'SVM': {
            'C': hp.loguniform('C', np.log(0.1), np.log(100)),
            'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'gamma': hp.choice('gamma', ['scale', 'auto'])
        },
        'KNN': {
            'n_neighbors': hp.randint('n_neighbors', 3, 20),
            'weights': hp.choice('weights', ['uniform', 'distance']),
            'metric': hp.choice('metric', ['euclidean', 'manhattan', 'minkowski'])
        },
        'XGBoost': {
            'n_estimators': hp.randint('n_estimators', 100, 300),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'max_depth': hp.randint('max_depth', 3, 7),
            'subsample': hp.uniform('subsample', 0.7, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0)
        }
    }

    # Lista para armazenar os resultados
    results = []

    # Treinamento e avaliação dos modelos usando execute_pipeline
    for model_name, model in models.items():
        config = 'CV'
        if enable_cv:
            print(f'Executing pipeline for {model_name} com cv apenas ...\n')
            report_cv, report_test = execute_pipeline_cv(model_name, model, preprocessor, X_train, y_train, X_test, y_test, label_encoder)
        if enable_gridsearch:
            print(f'Executing pipeline for {model_name} com GridSearch ...\n')
            report_cv, report_test = execute_pipeline_with_gridsearch(model_name, model, preprocessor, X_train, y_train, X_test, y_test, param_grids[model_name])
            config = 'GridSearch'
        if enable_randomsearch:
            print(f'Executing pipeline for {model_name} com RandomSearch ...\n')
            report_cv, report_test = execute_pipeline_with_randomsearch(model_name, model, preprocessor, X_train, y_train, X_test, y_test, param_distributions[model_name], n_iter=50)
            config = 'RandomSearch'
        if enable_bayesian_optimization:
            print(f'Executing pipeline for {model_name} com Bayesian Optimization ...\n')
            report_cv, report_test = execute_pipeline_with_bayesian_optimization(model_name, model, preprocessor, X_train, y_train, X_test, y_test, space[model_name], max_evals=50)
            config = 'BayesianOptimization'
        
        results.append({
        'Model': model_name,
        'Config': config,
        'Set': 'Train',
        **report_cv})
        
        results.append({
        'Model': model_name,
        'Config': config,
        'Set': 'Test',
        **report_test})

    return results

def execute_pipeline_cv (model_name, model, preprocessor, X_train, y_train, X_test, y_test, label_encoder):
    """
    Executa a pipeline para um único modelo, realizando validação cruzada e treinamento final.

    Args:
    model_name (str): Nome do modelo.
    model (sklearn estimator): Instância do modelo de aprendizado de máquina.
    preprocessor (ColumnTransformer): Pré-processador para as features.
    X_train (DataFrame): Dados de treinamento.
    y_train (Series): Rótulos de treinamento.
    X_test (DataFrame): Dados de teste.
    y_test (Series): Rótulos de teste.
    label_encoder (LabelEncoder): Codificador de rótulos.

    Returns:
    None
    """
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    # Realizar predições com validação cruzada
    y_pred_cv = cross_val_predict(clf, X_train, y_train, cv=5)
    
    print(f'Classification Report for {model_name} (Cross-Validation):\n')
    report_cv = classification_report(y_train, y_pred_cv, target_names=label_encoder.classes_, output_dict=True)
    print(report_cv)
    print('\n' + '='*80 + '\n')

    # Treinamento final e avaliação no conjunto de teste
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    
    print(f'Final Classification Report for {model_name} on Test Set:\n')
    report_test = classification_report(y_test, y_pred_test, target_names=label_encoder.classes_, output_dict=True)
    print(report_test)
    print('\n' + '='*80 + '\n')

    return report_cv, report_test

def generate_report_searchcv (search, y_train):
    # Calcular a média das previsões de todos os splits
    all_y_pred_cv = []
    for key in search.cv_results_:
        if key.startswith('split') and key.endswith('test_score'):  # Filtra as chaves relevantes
            all_y_pred_cv.append(search.cv_results_[key])  # Adiciona as previsões à lista

    # Calcula a média das previsões dos splits e arredonda para inteiros (classes)
    y_pred_cv_mean = np.mean(all_y_pred_cv, axis=0)  
    y_pred_cv_mean = np.rint(y_pred_cv_mean).astype(int) 

    # Gerar o relatório de classificação para a validação cruzada (usando a média)
    report_cv = classification_report(y_train, y_pred_cv_mean, output_dict=True)
    return report_cv

def execute_pipeline_with_gridsearch(model_name, model, preprocessor, X_train, y_train, X_test, y_test, param_grid):
    """
    Executa a pipeline com GridSearchCV para um único modelo.

    Args:
    model_name (str): Nome do modelo.
    model (sklearn estimator): Instância do modelo de aprendizado de máquina.
    preprocessor (ColumnTransformer): Pré-processador para as features.
    X_train (DataFrame): Dados de treinamento.
    y_train (Series): Rótulos de treinamento.
    X_test (DataFrame): Dados de teste.
    y_test (Series): Rótulos de teste.
    param_grid (dict): Grade de hiperparâmetros para GridSearchCV.

    Returns:
    None
    """
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)
    search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1, scoring=balanced_accuracy_scorer)
    # Ajustar GridSearchCV
    search.fit(X_train, y_train)
    
    report_cv = generate_report_searchcv (search, y_train, output_dict=True)
    # Imprimir os melhores parâmetros e o melhor score
    print(f'Best parameters for {model_name}: {search.best_params_}')
    print(f'Best cross-validation score for {model_name}: {search.best_score_}')
    
    # Avaliar no conjunto de teste
    y_pred_test = search.predict(X_test)
    print(f'Final Classification Report for {model_name} on Test Set:\n')
    report_test = classification_report(y_test, y_pred_test, output_dict=True)
    print(report_test)
    print('\n' + '='*80 + '\n')

    return report_cv, report_test


def execute_pipeline_with_randomsearch(model_name, model, preprocessor, X_train, y_train, X_test, y_test, param_distributions, n_iter=50):
    """
    Executa a pipeline com RandomizedSearchCV para um único modelo.

    Args:
    model_name (str): Nome do modelo.
    model (sklearn estimator): Instância do modelo de aprendizado de máquina.
    preprocessor (ColumnTransformer): Pré-processador para as features.
    X_train (DataFrame): Dados de treinamento.
    y_train (Series): Rótulos de treinamento.
    X_test (DataFrame): Dados de teste.
    y_test (Series): Rótulos de teste.
    param_distributions (dict): Distribuição de hiperparâmetros para RandomizedSearchCV.
    n_iter (int): Número de iterações para RandomizedSearchCV.

    Returns:
    None
    """
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    # Criar um "scorer" para acurácia balanceada
    balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)

    # Configurar RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_distributions, n_iter=n_iter, cv=5, verbose=1, n_jobs=-1, random_state=42, scoring=balanced_accuracy_scorer)

    # Ajustar RandomizedSearchCV
    random_search.fit(X_train, y_train)

    report_cv = generate_report_searchcv (random_search, y_train, output_dict=True)
    # Imprimir os melhores parâmetros e o melhor score
    print(f'Best parameters for {model_name}: {random_search.best_params_}')
    print(f'Best cross-validation score for {model_name}: {random_search.best_score_}')    
    
    # Avaliar no conjunto de teste
    y_pred_test = random_search.predict(X_test)
    print(f'Final Classification Report for {model_name} on Test Set:\n')
    report_test = classification_report(y_test, y_pred_test, output_dict=True)
    print(report_test)
    print('\n' + '='*80 + '\n')

    return report_cv, report_test  

def generate_report_bayesian(trials, y_train):
    """
    Gera um relatório de classificação a partir dos resultados da otimização Bayesiana (Hyperopt).

    Args:
        trials (hyperopt.Trials): Objeto Trials do Hyperopt contendo os resultados das avaliações.
        X_train (pd.DataFrame): DataFrame com os dados de treinamento.
        y_train (pd.Series): Série com os rótulos de treinamento.

    Returns:
        dict: Dicionário com o relatório de classificação (output_dict=True).
    """

    all_y_pred_cv = []

    # Itera sobre os trials (avaliações) do Hyperopt
    for trial in trials.trials:
        # Extrai as previsões de cada trial
        # ATENÇÃO: Adapte esta linha para a estrutura do seu código Hyperopt
        y_pred_cv = trial.result["y_pred"]  

        # Garante que as previsões sejam do tipo inteiro (classes)
        y_pred_cv = np.rint(y_pred_cv).astype(int)

        all_y_pred_cv.append(y_pred_cv)

    # Calcula a média das previsões de todos os trials
    y_pred_cv_mean = np.mean(all_y_pred_cv, axis=0)
    y_pred_cv_mean = np.rint(y_pred_cv_mean).astype(int)  # Arredonda para classes

    # Gera o relatório de classificação
    report_cv = classification_report(y_train, y_pred_cv_mean, output_dict=True)

    return report_cv


def execute_pipeline_with_bayesian_optimization(model_name, model, preprocessor, X_train, y_train, X_test, y_test, space, max_evals=50):
    """
    Executa a pipeline com otimização Bayesiana usando Hyperopt para um único modelo.

    Args:
    model_name (str): Nome do modelo.
    model (sklearn estimator): Instância do modelo de aprendizado de máquina.
    preprocessor (ColumnTransformer): Pré-processador para as features.
    X_train (DataFrame): Dados de treinamento.
    y_train (Series): Rótulos de treinamento.
    X_test (DataFrame): Dados de teste.
    y_test (Series): Rótulos de teste.
    space (dict): Espaço de hiperparâmetros para Hyperopt.
    max_evals (int): Número máximo de avaliações para Hyperopt.

    Returns:
    None
    """
    # Função objetivo para otimização Bayesiana
    def objective(params):
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model.set_params(**params))])
        balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)
        score = cross_val_score(clf, X_train, y_train, scoring=balanced_accuracy_scorer, cv=5).mean()
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, rstate=np.random.default_rng(42))

    print(f'Best parameters for {model_name}: {best_params}')

    report_cv = generate_report_bayesian(trials, y_train)
    
    # Configurar o modelo com os melhores parâmetros
    model.set_params(**best_params)
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    # Ajustar o modelo final
    clf.fit(X_train, y_train)
    
    # Avaliar no conjunto de teste
    y_pred_test = clf.predict(X_test)
    print(f'Final Classification Report for {model_name} on Test Set:\n')
    report_test = classification_report(y_test, y_pred_test, output_dict=True)
    print(report_test)    
    print('\n' + '='*80 + '\n')

    return report_cv, report_test