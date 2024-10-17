# %%
import os
path = "/Users/patricia/Documents/code/python-code/behavior-detection/src"
os.chdir(path)  # Muda o diretório para o nível anterior (a raiz do projeto)
print(os.getcwd())  # Verifique se agora está na raiz

# %% [markdown]
# # Load data

# %%
from behavior.behavior_data_loader import BehaviorDataLoader

data_path = '../data/new_logs_labels.csv'

data = BehaviorDataLoader.load_data(data_path, delimiter=';')
print(data.shape)
data.head(5)

# %%
from core.preprocessors.data_cleaner import DataCleaner

print("Valores da coluna 'comportamento' antes da remoção:", data['comportamento'].value_counts())

# Remove instances where 'comportamento' is '?'
data = DataCleaner.remove_instances_with_value(data, 'comportamento', '?')

print("\nValores da coluna 'comportamento' depois da remoção:", data['comportamento'].value_counts())

# %%
data.head(5)

# %%
## Select a subset of the data only for testing purposes

# Selecionar um subconjunto dos dados
# print("Tamanho do dataframe antes:", data.shape)
data = data.sample(n=500, random_state=42)
data.reset_index(drop=True, inplace=True)
# print("Tamanho do dataframe após:", data.shape)

# %% [markdown]
# # Pre-processing
# 
# ## Remove unnecessary columns

# %%
# Removing columns related to IDs, emotions, personality and behaviors, because 
# we want to classify behaviors only by the students' interactions with the system
columns_to_remove_ids = ['id_log', 'grupo', 'num_dia', 'num_log']
columns_to_remove_emotions = [
    'estado_afetivo', 'estado_engajamento_concentrado', 
    'estado_confusao', 'estado_frustracao', 'estado_tedio', 'estado_indefinido', 
    'ultimo_estado_afetivo', 'ultimo_engajamento_concentrado', 'ultimo_confusao', 
    'ultimo_frustracao', 'ultimo_tedio', 'ultimo_estado_indefinido'
]
columns_to_remove_personality = [
    'traco_amabilidade_fator', 'traco_extrovercao_fator', 'traco_conscienciosidade_fator', 
    'traco_abertura_fator', 'traco_neuroticismo_fator', 'traco_amabilidade_cat', 
    'traco_extrovercao_cat', 'traco_conscienciosidade_cat', 'traco_abertura_cat', 
    'traco_neuroticismo_cat']

columns_to_remove_behaviors = [
    'comportamento_on_task', 'comportamento_on_task_conversation', 'comportamento_on_task_out',
    'comportamento_off_task', 'comportamento_on_system', 'comportamento_indefinido',
    'ultimo_comportamento', 'ultimo_comportamento_on_task', 'ultimo_comportamento_on_task_conversation',
    'ultimo_comportamento_on_task_out', 'ultimo_comportamento_off_task', 'ultimo_comportamento_on_system',
    'ultimo_comportamento_indefinido'
]

columns_to_remove = columns_to_remove_ids + \
        columns_to_remove_emotions + \
        columns_to_remove_personality + \
        columns_to_remove_behaviors

cleaned_data = DataCleaner.remove_columns(data, columns_to_remove)


# %%
cleaned_data.head(5)

# %%
# Preenche valores ausentes no DataFrame X com a string 'missing'.

cleaned_data = cleaned_data.fillna('missing')

# %% [markdown]
# ## Split data by student level into training and test datasets

# %%
from core.preprocessors.data_splitter import DataSplitter

train_data, test_data = DataSplitter.split_by_student_level(cleaned_data, test_size=0.2, column_name='aluno')

# %%
# removing the 'aluno' column from the data after splitting into train and test sets

# Remover 'aluno' do conjunto de treinamento
cleaned_data = DataCleaner.remove_columns(train_data, ['aluno'])

# Remover 'aluno' do conjunto de teste
cleaned_data = DataCleaner.remove_columns(test_data, ['aluno'])

# %% [markdown]
# ## Split data into Features (X) and Target (y)

# %%
from core.preprocessors.data_splitter import DataSplitter

# Conjunto de treinamento
X_train, y_train = DataSplitter.split_into_x_y(train_data, 'comportamento')

# Conjunto de teste
X_test, y_test = DataSplitter.split_into_x_y(test_data, 'comportamento')

# %%
print("Primeiras 5 instâncias de y_train:")
print(y_train[:5])

print("\nPrimeiras 5 instâncias de y_test:")
print(y_test[:5])

# %% [markdown]
# ## Encoding variables

# %% [markdown]
# ### Encoding true labels (y)

# %%
import importlib
from core.preprocessors import column_selector, data_encoder
from behavior import behavior_data_encoder

# Recarregar o módulo para garantir que as alterações sejam aplicadas
importlib.reload(column_selector)
importlib.reload(data_encoder)
importlib.reload(behavior_data_encoder)

# %%
# Encoding y_train and y_test
from behavior.behavior_data_encoder import BehaviorDataEncoder

# Codificar y_train
y_train = BehaviorDataEncoder.encode_y(y_train)

# Codificar y_test
y_test = BehaviorDataEncoder.encode_y(y_test)

# %% [markdown]
# ### Encoding features (X)

# %%
# Pré-processar X_train
X_encoder = BehaviorDataEncoder(num_classes=5)
X_encoder.fit(X_train)

X_train = X_encoder.transform(X_train)

# Pré-processar X_test usando o mesmo preprocessor
X_test = X_encoder.transform(X_test)

# %% [markdown]
# # Balanceamento dos dados

# %%
from core.preprocessors.data_balancer import DataBalancer

data_balancer = DataBalancer()
X_train, y_train = data_balancer.apply_smote(X_train, y_train)

# %%
from collections import Counter

print(f"Resampled dataset shape: {Counter(y_train)}")

# %% [markdown]
# # Treinamento dos Modelos

# %% [markdown]
# ## Definindo parametros

# %%
# Definir quais modelos e seletores utilizar
selected_models = None # None to use all models
selected_selectors = ['pca', 'rf']

cv = 5  # Number of folds in the cross-validation
n_iter = 100
n_jobs = 4  # Number of processors to be used in the execution: -1 to use all processors

# Choose a scoring metric
scoring_metric = 'roc_auc_ovr'  # Possible values: 'f1_macro', 'balanced_accuracy', 'roc_auc_ovr', etc.

# %% [markdown]
# ## Usando Otimização Bayesiana (BayesSearchCV)

# %%
# from core.training.skopt_bayesian_optimization_training import SkoptBayesianOptimizationTraining

# training = SkoptBayesianOptimizationTraining()

# #### Executar o treinamento
# trained_models = training.train_model(
#     X_train=X_train,
#     y_train=y_train,
#     selected_models=selected_models,
#     selected_selectors=selected_selectors,
#     n_iter=n_iter,
#     cv=cv,
#     scoring=scoring_metric,
#     n_jobs=n_jobs
# )

# #### Exemplo de acesso aos modelos treinados
# for model_key, model_info in trained_models.items():
#     print(f"Modelo: {model_key}")
#     print(f"Melhores Hiperparâmetros: {model_info['hyperparameters']}")
#     print(f"Resultado CV: {model_info['cv_result']}\n") 

# %% [markdown]
# ## Usando Otimização Bayesiana (Optuna)

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
# # Avaliação dos Modelos

# %%
from core.evaluation.evaluation import Evaluation  

feature_names = X_train.columns  # Assumindo que os nomes das características são as colunas
class_metrics_results, avg_metrics_results = Evaluation.evaluate_all_models(trained_models, X_train, y_train, X_test, y_test, feature_names)

# %% [markdown]
# # Geração dos Relatórios

# %%
from core.logging.report_formatter import ReportFormatter
from core.logging.file_utils import FileUtils

directory = "../output/"

# Gerar relatório textual a partir dos resultados de avaliação
text_report = ReportFormatter.generate_text_report(class_metrics_results, avg_metrics_results)

# Imprimir ou salvar o relatório
FileUtils.save_file_with_timestamp(text_report, "bayesian_optimization_report.txt", directory)

# Gerar DataFrame detalhado dos relatórios por classe
class_report_df = ReportFormatter.generate_class_report_dataframe(class_metrics_results)

# Gerar DataFrame resumido dos relatórios de métricas médias
avg_metrics_report_df = ReportFormatter.generate_avg_metrics_report_dataframe(avg_metrics_results)

# Salvar os DataFrames como arquivos CSV, se necessário
FileUtils.save_csv_file_with_timestamp(class_report_df, "class_report.csv", directory)
FileUtils.save_csv_file_with_timestamp(avg_metrics_report_df, "avg_metrics_report.csv", directory)


# %% [markdown]
# 

# %% [markdown]
# # Salvando os modelos em arquivos para recuperação

# %%
from core.logging.model_manager import ModelManager

# Caminhos
model_dir = "../models/"

# Salvar todos os modelos
saved_models = ModelManager.save_all_models(trained_models, model_dir)
print("Modelos salvos:", saved_models)


