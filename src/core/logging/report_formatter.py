import pandas as pd

class ReportFormatter:
    @staticmethod
    def generate_text_report(class_metrics_reports, avg_metrics_reports):
        report_output = ""
        for model_name, model_info in class_metrics_reports.items():
            avg_info = avg_metrics_reports[model_name]
            report_output += ReportFormatter._format_model_report(model_name, model_info, avg_info)
        return report_output

    @staticmethod
    def _format_model_report(model_name, model_info, avg_info):
        report = (f"\nEvaluating {model_name} with {avg_info['training_type']}:\n"
                  f"Hyperparameters: {avg_info['hyperparameters']}\n")
        
        if 'selected_features' in model_info:
            report += f"\nSelected Features: {', '.join(str(feature) for feature in model_info['selected_features'])}\n"
        
        report += ReportFormatter._format_set_report("Train", model_info, avg_info)
        report += ReportFormatter._format_set_report("Test", model_info, avg_info)
        return report

    @staticmethod
    def _format_set_report(set_name, model_info, avg_info):
        report = f"\n{set_name} set class report:\n"
        report += ReportFormatter._format_classification_report(model_info[f'{set_name.lower()}_class_report'])
        report += f"\n{set_name} set average metrics:\n"
        report += ReportFormatter._format_classification_report(avg_info[f'{set_name.lower()}_avg_metrics'])
        report += f"\n{set_name} set confusion matrix:\n"
        report += model_info[f'{set_name.lower()}_conf_matrix'].to_string() + "\n"
        return report

    @staticmethod
    def _format_classification_report(report_df):
        output = ""
        for index, row in report_df.iterrows():
            if index in ['accuracy', 'balanced_accuracy']:
                output += f"{index.capitalize()}: {row['f1-score']}\n"
            else:
                output += (f"Class {index} - Precision: {row['precision']}, Recall: {row['recall']}, "
                           f"F1-Score: {row['f1-score']}, Support: {row['support']}\n")
        return output

    @staticmethod
    def generate_class_report_dataframe(class_metrics_reports):
        return pd.concat([
            ReportFormatter._prepare_model_report(model_name, model_info)
            for model_name, model_info in class_metrics_reports.items()
        ])

    @staticmethod
    def _prepare_model_report(model_name, model_info):
        train_report = model_info['train_class_report'].add_suffix('-train')
        test_report = model_info['test_class_report'].add_suffix('-test')
        combined_report = train_report.join(test_report)
        combined_report['Model'] = model_name
        return combined_report.reset_index().rename(columns={'index': 'Metric'})

    @staticmethod
    def generate_avg_metrics_report_dataframe(avg_metrics_reports):
        return pd.concat([
            ReportFormatter._prepare_avg_metrics_report(model_name, model_info)
            for model_name, model_info in avg_metrics_reports.items()
        ])

    @staticmethod
    def _prepare_avg_metrics_report(model_name, model_info):
        train_metrics = model_info['train_avg_metrics'].add_suffix('-train')
        test_metrics = model_info['test_avg_metrics'].add_suffix('-test')
        combined_report = train_metrics.join(test_metrics)
        combined_report['Model'] = model_name
        return combined_report.reset_index().rename(columns={'index': 'Metric'})