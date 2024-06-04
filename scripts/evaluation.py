import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import classification_report, balanced_accuracy_score

def evaluate_and_report(model, X, y):
    """Evaluates a model and returns a formatted report as a DataFrame."""
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    bal_acc = balanced_accuracy_score(y, y_pred)
    report['balanced_accuracy'] = {'precision': None, 'recall': None, 'f1-score': bal_acc, 'support': None}
    df = pd.DataFrame(report).transpose()
    df = df.round(2)
    return df

def generate_reports(trained_models, X_train, y_train, X_test, y_test):
    """Generates evaluation reports for both training and testing model_infosets for all models, including additional model information."""
    reports = {}
    for model_name, model_info in trained_models.items():
        # Extract the actual model from the dictionary
        model = model_info['model']
        train_df = evaluate_and_report(model, X_train, y_train)
        test_df = evaluate_and_report(model, X_test, y_test)
        
        # Storing both model_infoframes and additional info for later use
        reports[model_name] = {
            'train': train_df,
            'test': test_df,
            'training_type': model_info['training_type'],
            'hyperparameters': model_info['hyperparameters']
        }
    return reports

def print_reports(reports, directory=None, filename='report'):
    """Prints and optionally saves detailed evaluation reports for each model."""
    
    report_output = ""

    for model_name, model_info in reports.items():
        report_output += (f"\nEvaluating {model_name} with {model_info['training_type']}:\n"
                        f"Hiperparâmetros: {model_info['hyperparameters']}\n")
        for set_name, df in zip(['Training set', 'Test set'], [model_info['train'], model_info['test']]):
            report_output += f"\n{set_name} report:\n"
            for index, row in df.iterrows():
                if index == 'accuracy':
                    report_output += f"Accuracy - F1-Score: {row['f1-score']}\n"
                elif index == 'balanced_accuracy':
                    report_output += f"Balanced Accuracy: {row['f1-score']}\n"
                else:
                    report_output += f"Class {index} - Precision: {row['precision']}, Recall: {row['recall']}, F1-Score: {row['f1-score']}, Support: {row['support']}\n"


    # Formatando o nome do arquivo com a model_info e hora atual
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.txt"
    file_path = os.path.join(directory, filename) if directory else filename

    if directory:
        with open(file_path, 'w') as file:
            file.write(report_output)
            file.flush()
            os.fsync(file.fileno())

    return report_output

def save_reports_to_csv(reports, directory, detailed_filename='detailed_report', summary_filename='summary_report'):
    """Saves detailed and summary classification reports to CSV files."""

    all_model_info = []
    summary_model_info = []
    for model_name, model_info in reports.items():
        # Rename columns to indicate train/test and merge model_info
        model_info['train'].columns = [f'{col}-train' for col in model_info['train'].columns]
        model_info['test'].columns = [f'{col}-test' for col in model_info['test'].columns]
        report_df = model_info['train'].join(model_info['test'])
        
        # Clean up 'accuracy' rows
        if 'accuracy' in report_df.index:
            accuracy_cols = [col for col in report_df.columns if 'f1-score' not in col]
            report_df.loc['accuracy', accuracy_cols] = None
        
        # Add model and reset index
        report_df['Model'] = model_name
        report_df.reset_index(inplace=True)
        report_df.rename(columns={'index': 'Metric'}, inplace=True)

        all_model_info.append(report_df)
        summary_model_info.append(report_df[report_df['Metric'].isin(['accuracy', 'macro avg', 'weighted avg', 'balanced_accuracy'])])


    # Formatando o nome do arquivo com a model_info e hora atual
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_filename = f"{detailed_filename}_{timestamp}.csv"
    summary_filename = f"{summary_filename}_{timestamp}.csv"

    pd.concat(all_model_info).to_csv(os.path.join(directory, detailed_filename), index=False)
    pd.concat(summary_model_info).to_csv(os.path.join(directory, summary_filename), index=False)

