# %% [markdown]
# ## Imports

# %%
import os
path = "/Users/patricia/Documents/code/python-code/behavior-detection/src"
os.chdir(path)  # Muda o diretório para o nível anterior (a raiz do projeto)
print(os.getcwd())  # Verifique se agora está na raiz

# %%
from sklearn import datasets
from time import time
import pandas as pd

# %% [markdown]
# ## Data

# %%
digits = datasets.load_digits()

n_samples = len(digits.images)
X = digits.images
y = digits.target

# Flatten the images
X = X.reshape((n_samples, -1))
X = pd.DataFrame(X)

# %%
print(n_samples)
print(type(X))
print(X.head(5))
print(y[:5])

# %%
from sklearn.model_selection import train_test_split

# Dividir X e y em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ## Preprocessing

# %%
import pandas as pd
from core.preprocessors.data_encoder import DataEncoder

# Codificar y_train
y_train = DataEncoder.encode_y(y_train)
# Codificar y_test
y_test = DataEncoder.encode_y(y_test)

# Pré-processar X_train
X_encoder = DataEncoder(num_classes=0, select_numerical=True)
X_encoder.fit(X_train)

X_train = X_encoder.transform(X_train)

# Pré-processar X_test usando o mesmo preprocessor
X_test = X_encoder.transform(X_test)

# %%
print(X_train.head(15))

# %% [markdown]
# ## Classifier

# %%
# Definir quais modelos e seletores utilizar
selected_models = [ 
    # 'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'Gradient Boosting',
    'SVM',
    'KNN',
    'XGBoost'
]
selected_selectors = ['pca', 'rf']

cv = 5  # Number of folds in the cross-validation
n_iter = 100
n_jobs = 4  # Number of processors to be used in the execution: -1 to use all processors

# Choose a scoring metric
scoring_metric = 'balanced_accuracy'  # Possible values: 'f1_macro', 'balanced_accuracy', 'roc_auc_ovr', etc.

# %% [markdown]
# ## Usando Optuna (Otimização Bayesiana)

# %%
# Importação da nova classe OptunaBayesianOptimizationTraining
from core.training.optuna_bayesian_optimization_training import OptunaBayesianOptimizationTraining

# Instanciação da classe de treinamento com Otimização Bayesiana via Optuna
training = OptunaBayesianOptimizationTraining()

# Executar o treinamento
trained_models = training.train_model(
    X_train=X_train,
    y_train=y_train,
    selected_models=selected_models,
    selected_selectors=selected_selectors,
    n_iter=n_iter,  # Será mapeado para n_trials na classe OptunaBayesianOptimizationTraining
    cv=cv,
    scoring=scoring_metric,
    n_jobs=n_jobs
)

# Exemplo de acesso aos modelos treinados
for model_key, model_info in trained_models.items():
    print(f"Modelo: {model_key}")
    print(f"Melhores Hiperparâmetros: {model_info['hyperparameters']}")
    print(f"Resultado CV: {model_info['cv_result']}\n")


# %% [markdown]
# ### Avaliação e logging

# %%
import notebook_utils as nb_utils

# Avaliação dos Modelos
class_metrics_results, avg_metrics_results = nb_utils.evaluate_models(trained_models, X_train, y_train, X_test, y_test)

# Geração dos Relatórios
nb_utils.generate_reports(class_metrics_results, avg_metrics_results, filename_prefix="_Optuna_")

# Salvando os modelos em arquivos para recuperação
nb_utils.save_models(trained_models, filename_prefix="_Optuna_")

# %% [markdown]
# ## Usando BayesSearchCV (Otimização Bayesiana)

# %%
from core.training.skopt_bayesian_optimization_training import SkoptBayesianOptimizationTraining

training = SkoptBayesianOptimizationTraining()

#### Executar o treinamento
trained_models = training.train_model(
    X_train=X_train,
    y_train=y_train,
    selected_models=selected_models,
    selected_selectors=selected_selectors,
    n_iter=n_iter,
    cv=cv,
    scoring=scoring_metric,
    n_jobs=n_jobs
)

#### Exemplo de acesso aos modelos treinados
for model_key, model_info in trained_models.items():
    print(f"Modelo: {model_key}")
    print(f"Melhores Hiperparâmetros: {model_info['hyperparameters']}")
    print(f"Resultado CV: {model_info['cv_result']}\n") 

# %% [markdown]
# ### Avaliação e logging

# %%
import notebook_utils as nb_utils

# Avaliação dos Modelos
class_metrics_results, avg_metrics_results = nb_utils.evaluate_models(trained_models, X_train, y_train, X_test, y_test)

# Geração dos Relatórios
nb_utils.generate_reports(class_metrics_results, avg_metrics_results, filename_prefix="_Skopt_")

# Salvando os modelos em arquivos para recuperação
nb_utils.save_models(trained_models, filename_prefix="_Skopt_")

# %% [markdown]
# ## Using GridSearchCV

# %%
from core.training.grid_search_training import GridSearchTraining

# Instantiate the GridSearchCV training class
training = GridSearchTraining()

# Execute the training
trained_models = training.train_model(
    X_train=X_train,
    y_train=y_train,
    selected_models=selected_models,
    selected_selectors=selected_selectors,
    n_iter=n_iter,  # This parameter is not used in GridSearchCV but kept for consistency
    cv=cv,
    scoring=scoring_metric,
    n_jobs=n_jobs
)

# Example of accessing the trained models
for model_key, model_info in trained_models.items():
    print(f"Model: {model_key}")
    print(f"Best Hyperparameters: {model_info['hyperparameters']}")
    print(f"CV Result: {model_info['cv_result']}\n")

# %% [markdown]
# ### Avaliação e logging

# %%
import notebook_utils as nb_utils

# Avaliação dos Modelos
class_metrics_results, avg_metrics_results = nb_utils.evaluate_models(trained_models, X_train, y_train, X_test, y_test)

# Geração dos Relatórios
nb_utils.generate_reports(class_metrics_results, avg_metrics_results, filename_prefix="_GridSearch_")

# Salvando os modelos em arquivos para recuperação
nb_utils.save_models(trained_models, filename_prefix="_GridSearch_")

# %% [markdown]
# ## Treinando com RandomSearchCV

# %%
# src/notebooks/model_training_behavior_multiclassification_by_student_level.ipynb

from core.training.random_search_training import RandomSearchTraining

# Instantiate the RandomizedSearchCV training class
training = RandomSearchTraining()

# Execute the training
trained_models = training.train_model(
    X_train=X_train,
    y_train=y_train,
    selected_models=selected_models,
    selected_selectors=selected_selectors,
    n_iter=n_iter,
    cv=cv,
    scoring=scoring_metric,
    n_jobs=n_jobs
)

# Example of accessing the trained models
for model_key, model_info in trained_models.items():
    print(f"Model: {model_key}")
    print(f"Best Hyperparameters: {model_info['hyperparameters']}")
    print(f"CV Result: {model_info['cv_result']}\n")

# %% [markdown]
# ### Avaliação e logging

# %%
import notebook_utils as nb_utils

# Avaliação dos Modelos
class_metrics_results, avg_metrics_results = nb_utils.evaluate_models(trained_models, X_train, y_train, X_test, y_test)

# Geração dos Relatórios
nb_utils.generate_reports(class_metrics_results, avg_metrics_results, filename_prefix="_RandomSearch_")

# Salvando os modelos em arquivos para recuperação
nb_utils.save_models(trained_models, filename_prefix="_RandomSearch_")


