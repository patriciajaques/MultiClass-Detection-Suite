#!/usr/bin/env python
# coding: utf-8



import preprocessing as pre


# No notebook
data_path = 'data/new_logs_labels.csv'
X, y = pre.load_data(data_path)
y.head()





X = X.fillna('missing')





test_size = 0.2  # 80% for training, 20% for testing
X_train, X_test, y_train, y_test = pre.split_train_test_data(X, y, test_size, random_state=42)





import pandas as pd

print("Nro de instancias de cada classe em y_train:\n")
print(pd.Series(y_train).value_counts())
print("\n\nNro de instancias de cada classe em y_test:\n")
print(pd.Series(y_test).value_counts())





non_numeric_cols_train = X_train.select_dtypes(exclude=['float', 'int']).columns
non_numeric_cols_test = X_test.select_dtypes(exclude=['float', 'int']).columns

print("Non-numeric columns in X_train:")
print(non_numeric_cols_train)

print("\nNon-numeric columns in X_test:")
print(non_numeric_cols_test)





X_train, label_encoders = pre.encode_categorical_columns(X_train)
X_test = pre.apply_encoders_to_test_data(X_test, label_encoders)





import pandas as pd

print("Nro de instancias de cada classe em y_train:\n")
print(pd.Series(y_train).value_counts())
print("\n\nNro de instancias de cada classe em y_test:\n")
print(pd.Series(y_test).value_counts())





y_train, label_encoder = pre.encode_single_column(y_train)
y_test = label_encoder.transform(y_test)





X_train_over, y_train_over = pre.apply_smote(X_train, y_train)





import pandas as pd

print("Nro de instancias de cada classe em y_train:\n")
print(pd.Series(y_train_over).value_counts())





# Visualizar os tipos das colunas de X_train_over
print("Tipos das colunas de X_train_over:")
x_train_types = X_train_over.dtypes

# Visualizar os tipos das colunas de X_test
print("\nTipos das colunas de X_test:")
X_test_types = X_test.dtypes





# Pré-processar os dados uma vez
preprocessor = pre.create_preprocessor(X_train_over)
X_train_preprocessed = preprocessor.fit_transform(X_train_over)
X_test_preprocessed = preprocessor.transform(X_test)





print(X_train_preprocessed[:5])





print(X_train_preprocessed.shape)
import pandas as pd

# Supondo que 'X_train_preprocessed' seja seu numpy.ndarray
df = pd.DataFrame(X_train_preprocessed)

# Agora você pode chamar .describe() no DataFrame
print(df.describe())





# Geração dos relatórios
feature_names = X_train.columns  # Assumindo que os nomes das características são as colunas
print("feature_names: ", feature_names)





# Verificação antes de chamar a função de treinamento
print(f"X_train_preprocessed shape: {X_train_preprocessed.shape}")
print(f"y_train shape: {y_train_over.shape}")


# # Treinamento dos Modelos usando Otimização Bayesiana (BayesSearchCV)




from bayesian_optimization_training import BayesianOptimizationTraining
cv = 10
n_iter = 100

# Escolher a métrica de avaliação
scoring_metric = 'roc_auc_ovr'  # Pode ser 'f1_macro', 'balanced_accuracy', 'roc_auc_ovr', etc.


training = BayesianOptimizationTraining()


# Chamar o treinamento com otimização bayesiana
trained_models = training.train_model(
    X_train_preprocessed, y_train_over, n_iter=n_iter, cv=cv, scoring=scoring_metric
)


# # Avaliação dos Modelos


from evaluation import Evaluation 

# Geração dos relatórios
feature_names = X_train.columns  # Assumindo que os nomes das características são as colunas
class_metrics_results, avg_metrics_results = Evaluation.evaluate_all_models(trained_models, X_train_preprocessed, y_train_over, X_test_preprocessed, y_test, feature_names)



# # Geração dos Relatórios

from report_formatter import ReportFormatter
from file_manager import FileManager

directory = "../output/"

# Geração dos relatórios
feature_names = X_train.columns  # Assumindo que os nomes das características são as colunas
class_metrics_results, avg_metrics_results = Evaluation.evaluate_all_models(trained_models, X_train_preprocessed, y_train_over, X_test_preprocessed, y_test, feature_names)

# Impressão dos relatórios

# Gerar relatório textual a partir dos resultados de avaliação
text_report = ReportFormatter.generate_text_report_from_dict(class_metrics_results, avg_metrics_results)

# Imprimir ou salvar o relatório
FileManager.save_text_file_with_timestamp(text_report, "bayesian_optimization_report.txt", directory)

# Gerar DataFrame detalhado dos relatórios por classe
class_report_df = ReportFormatter.generate_class_report_dataframe(class_metrics_results)

# Gerar DataFrame resumido dos relatórios de métricas médias
avg_metrics_report_df = ReportFormatter.generate_avg_metrics_report_dataframe(avg_metrics_results)

# Salvar os DataFrames como arquivos CSV, se necessário
FileManager.save_csv_file_with_timestamp(class_report_df, "class_report.csv", directory)
FileManager.save_csv_file_with_timestamp(avg_metrics_report_df, "avg_metrics_report.csv", directory)


# # Salvando os modelos em arquivos para recuperação




from model_manager import ModelManager

# Caminhos
model_dir = "../models/"

# Salvar todos os modelos
saved_models = ModelManager.dump_all_models(trained_models, model_dir)
print("Modelos salvos:", saved_models)

