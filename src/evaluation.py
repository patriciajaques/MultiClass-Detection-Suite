import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
import joblib
from feature_selection import extract_selected_features

def generate_reports(trained_models, X_train, y_train, X_test, y_test, feature_names=None):
    """
    Gera relatórios de avaliação para todos os modelos nos conjuntos de treino e teste, utilizando os resultados da
    validação cruzada realizada pelo BayesSearchCV.

    Args:
        trained_models: Dicionário de modelos treinados com BayesSearchCV.
        X_train: Conjunto de características de treino.
        y_train: Conjunto de rótulos de treino.
        X_test: Conjunto de características de teste.
        y_test: Conjunto de rótulos de teste.
        feature_names: Lista de nomes das características (opcional).

    Returns:
        reports: Dicionário com os relatórios de avaliação.
    """
    reports = {}
    for model_name, model_info in trained_models.items():
        model = model_info['model']

        # Acessar os resultados da validação cruzada do BayesSearchCV
        cv_scores = model.cv_results_['mean_test_score']
        cv_report = {'mean_score': np.mean(cv_scores), 'std_score': np.std(cv_scores)}

        # Avaliar o modelo no conjunto de treinamento usando predict
        train_report, train_conf_matrix, train_selected_features = evaluate_model(model, X_train, y_train, feature_names)

        # Avaliar o modelo no conjunto de teste
        test_report, test_conf_matrix, test_selected_features = evaluate_model(model, X_test, y_test, feature_names)
        
        reports[model_name] = {
            'cv_report': cv_report,
            'train_report': train_report,
            'train_conf_matrix': train_conf_matrix,
            'test_report': test_report,
            'test_conf_matrix': test_conf_matrix,
            'training_type': model_info['training_type'],
            'hyperparameters': model.best_params_,
            'selected_features': test_selected_features
        }
    return reports

def evaluate_model(model, X, y, feature_names=None):
    """
    Avalia um modelo e retorna as previsões, as métricas de avaliação e as características selecionadas.

    Args:
        model: Modelo treinado.
        X: Conjunto de características.
        y: Conjunto de rótulos.
        feature_names: Lista de nomes das características (opcional).

    Returns:
        report_df: DataFrame contendo o relatório de classificação.
        conf_matrix_df: DataFrame contendo a matriz de confusão.
        selected_features: Lista de características selecionadas (se aplicável).
    """
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    bal_acc = balanced_accuracy_score(y, y_pred)
    report['balanced_accuracy'] = {'precision': None, 'recall': None, 'f1-score': bal_acc, 'support': None}
    report_df = pd.DataFrame(report).transpose().round(2)
    
    conf_matrix = confusion_matrix(y, y_pred)
    conf_matrix_df = pd.DataFrame(conf_matrix, 
                                  index=[f'Actual {cls}' for cls in sorted(set(y))], 
                                  columns=[f'Predicted {cls}' for cls in sorted(set(y))])

    if feature_names is not None:
        selected_features = extract_selected_features(model, feature_names)
    else:
        selected_features = None
    
    return report_df, conf_matrix_df, selected_features

def format_report(report_df):
    """
    Formata o relatório de classificação como uma string.
    
    Returns:
        output: String formatada do relatório de classificação.
    """
    output = ""
    for index, row in report_df.iterrows():
        if index == 'accuracy':
            output += f"Accuracy - F1-Score: {row['f1-score']}\n"
        elif index == 'balanced_accuracy':
            output += f"Balanced Accuracy: {row['f1-score']}\n"
        else:
            output += (f"Class {index} - Precision: {row['precision']}, Recall: {row['recall']}, "
                       f"F1-Score: {row['f1-score']}, Support: {row['support']}\n")
    return output

def save_text_file(content, filename, directory=None):
    """
    Salva o conteúdo em um arquivo de texto.
    
    Returns:
        file_path: Caminho completo do arquivo salvo.
    """
    if directory:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, filename)
    else:
        file_path = filename

    with open(file_path, 'w') as file:
        file.write(content)
    
    return file_path

def save_csv_file(dataframe, filename, directory=None):
    """
    Salva o DataFrame em um arquivo CSV.
    
    Returns:
        file_path: Caminho completo do arquivo salvo.
    """
    if directory:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, filename)
    else:
        file_path = filename
    
    dataframe.to_csv(file_path, index=False)
    return file_path

def print_reports(reports, directory=None, filename='report'):
    """
    Imprime e opcionalmente salva relatórios detalhados de avaliação, incluindo matrizes de confusão.

    Args:
        reports: Dicionário com os relatórios de avaliação.
        directory: Diretório para salvar os relatórios.
        filename: Nome do arquivo para salvar os relatórios.

    Returns:
        report_output: String com o conteúdo dos relatórios.
    """
    report_output = ""

    for model_name, model_info in reports.items():
        report_output += (f"\nEvaluating {model_name} with {model_info['training_type']}:\n"
                          f"Hyperparameters: {model_info['hyperparameters']}\n")
        
        selected_features = model_info.get('selected_features', None)
        if selected_features is not None:
            report_output += f"\nSelected Features: {', '.join(selected_features)}\n"
        
        for eval_type, set_name, report, conf_matrix in zip(
                ['Cross-Validation', 'Training', 'Test'],
                ['cross-validation', 'training', 'test'],
                [model_info['cv_report'], model_info['train_report'], model_info['test_report']],
                [model_info['cv_conf_matrix'], model_info['train_conf_matrix'], model_info['test_conf_matrix']]
            ):
            report_output += f"\n{eval_type} set report:\n"
            report_output += format_report(report)
            report_output += f"\n{set_name.capitalize()} set confusion matrix:\n"
            report_output += conf_matrix.to_string() + "\n"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{filename}_{timestamp}.txt"
    save_text_file(report_output, filename, directory)
    return report_output

def save_reports_to_csv(reports, directory, detailed_filename='detailed_report', summary_filename='summary_report'):
    """
    Salva relatórios detalhados e resumidos de classificação em arquivos CSV.
    """
    all_model_info = []
    summary_model_info = []

    for model_name, model_info in reports.items():
        train_report = model_info['train_report'].add_suffix('-train')
        test_report = model_info['test_report'].add_suffix('-test')
        combined_report = train_report.join(test_report)
        
        if 'accuracy' in combined_report.index:
            accuracy_cols = [col for col in combined_report.columns if 'f1-score' not in col]
            combined_report.loc['accuracy', accuracy_cols] = None

        combined_report['Model'] = model_name
        combined_report.reset_index(inplace=True)
        combined_report.rename(columns={'index': 'Metric'}, inplace=True)

        all_model_info.append(combined_report)
        summary_model_info.append(combined_report[combined_report['Metric'].isin(['accuracy', 'macro avg', 'weighted avg', 'balanced_accuracy'])])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    detailed_filename = f"{detailed_filename}_{timestamp}.csv"
    summary_filename = f"{summary_filename}_{timestamp}.csv"

    save_csv_file(pd.concat(all_model_info), detailed_filename, directory)
    save_csv_file(pd.concat(summary_model_info), summary_filename, directory)

def dump_model(model, filename, directory=None):
    """
    Salva o modelo treinado em um arquivo.
    
    Returns:
        file_path: Caminho completo do arquivo salvo.
    """
    if directory:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, filename)
    else:
        file_path = filename
    
    joblib.dump(model, file_path)
    return file_path

def load_model(filename, directory=None):
    """
    Carrega um modelo salvo a partir de um arquivo.
    
    Returns:
        model: Modelo carregado.
    """
    if directory:
        file_path = os.path.join(directory, filename)
    else:
        file_path = filename
    
    model = joblib.load(file_path)
    return model

def dump_all_models(trained_models, directory, prefix='model'):
    """
    Salva todos os modelos treinados em arquivos individuais com data e hora.
    
    Returns:
        saved_models: Lista de caminhos completos dos arquivos salvos.
    """
    saved_models = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    for model_name, model_info in trained_models.items():
        model = model_info['model']
        filename = f"{prefix}_{model_name}_{timestamp}.pkl"
        file_path = dump_model(model, filename, directory)
        saved_models.append(file_path)
        print(f"Modelo '{model_name}' salvo em: {file_path}")
    
    return saved_models