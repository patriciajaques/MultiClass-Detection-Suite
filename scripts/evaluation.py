import pandas as pd
import os  # Importing os to handle path operations
from sklearn.metrics import classification_report, balanced_accuracy_score

def evaluate_and_report(model, X, y):
    """Helper function to evaluate the model and return a formatted report DataFrame."""
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    bal_acc = balanced_accuracy_score(y, y_pred)
    report['balanced accuracy'] = {'precision': None, 'recall': None, 'f1-score': bal_acc, 'support': None}
    df = pd.DataFrame(report).transpose()
    df = df.round(2)  # Round all numeric values to two decimal places
    return df

def save_reports_to_csv(results, X_train, X_test, y_train, y_test, directory):
    """Generates and saves detailed and summary classification reports to CSV files."""
    all_data = []
    summary_data = []

    for model_name, model_info in results.items():
        model = model_info['model']
        # Generate and format reports
        train_df = evaluate_and_report(model, X_train, y_train)
        test_df = evaluate_and_report(model, X_test, y_test)
        
        # Rename columns to indicate train/test and merge data
        train_df.columns = [f'{col}-train' for col in train_df.columns]
        test_df.columns = [f'{col}-test' for col in test_df.columns]
        report_df = train_df.join(test_df)
        
        # Clean up 'accuracy' rows
        if 'accuracy' in report_df.index:
            accuracy_cols = [col for col in report_df.columns if 'f1-score' not in col]
            report_df.loc['accuracy', accuracy_cols] = None
        
        # Add model and reset index
        report_df['Model'] = model_name
        report_df.reset_index(inplace=True)
        report_df.rename(columns={'index': 'Metric'}, inplace=True)

        all_data.append(report_df)
        # Extract summary metrics separately
        summary_metrics = ['accuracy', 'macro avg', 'weighted avg', 'balanced accuracy']
        summary_df = report_df[report_df['Metric'].isin(summary_metrics)]
        summary_data.append(summary_df)

    # Construct file paths
    detailed_file_path = os.path.join(directory, 'detailed_report.csv')
    summary_file_path = os.path.join(directory, 'summary_report.csv')

    # Construct file paths
    detailed_file_path = os.path.join(directory, 'detailed_report.csv')
    summary_file_path = os.path.join(directory, 'summary_report.csv')

    # Save detailed and summary CSVs
    pd.concat(all_data).to_csv(detailed_file_path, index=False)
    pd.concat(summary_data).to_csv(summary_file_path, index=False)


def print_reports(results, X_train, X_test, y_train, y_test, directory):
    """Prints and saves detailed evaluation reports for each model to a specified text file."""
    file_path = os.path.join(directory, 'evaluation_report.txt')
    with open(file_path, 'w') as file:
        for model_name, model_info in results.items():
            best_model = model_info['model']
            report_output = f"\nEvaluating {model_name}:\n"

            datasets = {'Training set': (X_train, y_train), 'Test set': (X_test, y_test)}
            for set_name, (X, y) in datasets.items():
                report = evaluate_and_report(best_model, X, y)
                report_output += f"\n{set_name} report:\n"
                # Iterate over rows in the report DataFrame
                for index, row in report.iterrows():
                    if index in ['macro avg', 'weighted avg', 'balanced_accuracy']:
                        prefix = "Global " if index != 'balanced_accuracy' else "Global Balanced "
                        report_output += f"{prefix}{index.capitalize()} - Precision: {row['precision']}, Recall: {row['recall']}, F1-Score: {row['f1-score']}, Support: {row['support']}\n"
                    elif index == 'accuracy':
                        report_output += f"Global Accuracy - F1-Score: {row['f1-score']}\n"
                    else:
                        report_output += f"Class {index} - Precision: {row['precision']}, Recall: {row['recall']}, F1-Score: {row['f1-score']}, Support: {row['support']}\n"

            # Print to console
            print(report_output)
            # Write to file
            file.write(report_output)

