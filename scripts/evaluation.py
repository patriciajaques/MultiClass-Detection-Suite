import pandas as pd
from sklearn.metrics import classification_report, balanced_accuracy_score

def evaluate_model(model, X, y):
    """Predicts using the model and returns a detailed classification report."""
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    # Calculate balanced accuracy and add it to the report
    bal_acc = balanced_accuracy_score(y, y_pred)
    report['balanced_accuracy'] = {'precision': None, 'recall': None, 'f1-score': bal_acc, 'support': None}
    return report

def save_reports_to_csv(results, X_train, X_test, y_train, y_test, filename='model_evaluation_reports.csv'):
    """Generates and saves classification reports to a CSV file, including class names."""
    report_data = []

    for model_name, model_info in results.items():
        best_model = model_info['model']
        # Generate reports for train and test datasets
        train_report = pd.DataFrame(evaluate_model(best_model, X_train, y_train)).transpose()
        test_report = pd.DataFrame(evaluate_model(best_model, X_test, y_test)).transpose()

        # Prepare train and test reports
        train_report.columns = [f"{col}-train" for col in train_report.columns]
        test_report.columns = [f"{col}-test" for col in test_report.columns]

        # Format numeric values to two decimal places
        for df in [train_report, test_report]:
            for col in df.columns:
                if 'f1-score' in col or 'precision' in col or 'recall' in col:
                    df[col] = df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)

        # Merge train and test reports on the index, which contains the class labels
        merged_report = train_report.join(test_report, how='outer')

        # Add model and set information
        merged_report['Model'] = model_name
        merged_report.reset_index(inplace=True)
        merged_report.rename(columns={'index': 'Class'}, inplace=True)

        report_data.append(merged_report)

    # Concatenate all model data into a single DataFrame
    final_df = pd.concat(report_data, ignore_index=True)
    final_df.to_csv(filename, index=False)


def print_reports(results, X_train, X_test, y_train, y_test):
    """Prints detailed evaluation reports for each model."""
    for model_name, model_info in results.items():
        best_model = model_info['model']
        print(f"Evaluating {model_name}:")

        datasets = {'Training set': (X_train, y_train), 'Test set': (X_test, y_test)}
        for set_name, (X, y) in datasets.items():
            report = evaluate_model(best_model, X, y)
            print(f"{set_name} report:")
            # Print each class report formatted
            for key, values in report.items():
                if isinstance(values, dict):
                    # Safely format numbers or handle None
                    precision = f"{values['precision']:.2f}" if values['precision'] is not None else 'None'
                    recall = f"{values['recall']:.2f}" if values['recall'] is not None else 'None'
                    f1_score = f"{values['f1-score']:.2f}" if values['f1-score'] is not None else 'None'
                    support = values['support'] if values['support'] is not None else 'None'
                    print(f"Class {key} - Precision: {precision}, Recall: {recall}, F1-Score: {f1_score}, Support: {support}")
            # Print balanced accuracy separately
            bal_acc = f"{report['balanced_accuracy']['f1-score']:.2f}" if report['balanced_accuracy']['f1-score'] is not None else 'None'
            print(f"Balanced Accuracy: {bal_acc}")
