import os
import pandas as pd
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
    """Generates evaluation reports for both training and testing datasets for all models."""
    reports = {}
    for model_name, model in trained_models.items():
        train_df = evaluate_and_report(model, X_train, y_train)
        test_df = evaluate_and_report(model, X_test, y_test)
        reports[model_name] = {'train': train_df, 'test': test_df}
    return reports

def print_reports(reports, directory=None, filename='report.txt'):
    """Prints and optionally saves detailed evaluation reports for each model."""
    report_output = ""
    for model_name, data in reports.items():
        report_output += f"\nEvaluating {model_name}:\n"
        for set_name, df in zip(['Training set', 'Test set'], [data['train'], data['test']]):
            report_output += f"\n{set_name} report:\n"
            for index, row in df.iterrows():
                if index == 'accuracy':
                    report_output += f"Accuracy - F1-Score: {row['f1-score']}\n"
                elif index == 'balanced_accuracy':
                    report_output += f"Balanced Accuracy: {row['f1-score']}\n"
                else:
                    report_output += f"Class {index} - Precision: {row['precision']}, Recall: {row['recall']}, F1-Score: {row['f1-score']}, Support: {row['support']}\n"

    if directory:
        file_path = os.path.join(directory, filename)
        with open(file_path, 'w') as file:
            file.write(report_output)

    return report_output

def save_reports_to_csv(reports, directory, detailed_filename='detailed_report.csv', summary_filename='summary_report.csv'):
    """Saves detailed and summary classification reports to CSV files."""
    all_data = []
    summary_data = []
    for model_name, data in reports.items():
        # Rename columns to indicate train/test and merge data
        data['train'].columns = [f'{col}-train' for col in data['train'].columns]
        data['test'].columns = [f'{col}-test' for col in data['test'].columns]
        report_df = data['train'].join(data['test'])
        
        # Clean up 'accuracy' rows
        if 'accuracy' in report_df.index:
            accuracy_cols = [col for col in report_df.columns if 'f1-score' not in col]
            report_df.loc['accuracy', accuracy_cols] = None
        
        # Add model and reset index
        report_df['Model'] = model_name
        report_df.reset_index(inplace=True)
        report_df.rename(columns={'index': 'Metric'}, inplace=True)

        all_data.append(report_df)
        summary_data.append(report_df[report_df['Metric'].isin(['accuracy', 'macro avg', 'weighted avg', 'balanced_accuracy'])])

    pd.concat(all_data).to_csv(os.path.join(directory, detailed_filename), index=False)
    pd.concat(summary_data).to_csv(os.path.join(directory, summary_filename), index=False)

