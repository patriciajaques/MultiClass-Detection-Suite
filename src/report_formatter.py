import pandas as pd

class ReportFormatter:

    @staticmethod
    def generate_text_report_from_dict(reports):
        """
        Gera texto para relatórios detalhados de avaliação, incluindo matrizes de confusão.

        Args:
            reports: Dicionário com os relatórios de avaliação.

        Returns:
            report_output: String com o conteúdo dos relatórios.
        """
        report_output = ""

        for model_name, model_info in reports.items():
            report_output += (f"\nEvaluating {model_name} with {model_info['training_type']}:\n"
                                f"Hyperparameters: {model_info['hyperparameters']}\n")
            
            selected_features = model_info.get('selected_features', None)
            if selected_features is not None:
                report_output += f"\nSelected Features: {', '.join(str(feature) for feature in selected_features)}\n"
            
            # Relatório do conjunto de treino
            report_output += "\nTrain set report:\n"
            report_output += ReportFormatter.format_report(model_info['train_report'])
            report_output += "\nTrain set confusion matrix:\n"
            report_output += model_info['train_conf_matrix'].to_string() + "\n"
            
            # Relatório do conjunto de teste
            report_output += "\nTest set report:\n"
            report_output += ReportFormatter.format_report(model_info['test_report'])
            report_output += "\nTest set confusion matrix:\n"
            report_output += model_info['test_conf_matrix'].to_string() + "\n"

        return report_output
    
    @staticmethod
    def format_report(report_df):
        """
        Formata o relatório de classificação como uma string.
        
        Args:
            report_df: DataFrame contendo o relatório de classificação.

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

    @staticmethod
    def generate_detailed_report_dataframe(reports):
        """
        Gera um DataFrame detalhado de classificação.
        """
        all_model_info = []

        for model_name, model_info in reports.items():
            train_report = model_info['train_report'].add_suffix('-train')
            test_report = model_info['test_report'].add_suffix('-test')
            combined_report = train_report.join(test_report)
            
            combined_report['Model'] = model_name
            combined_report.reset_index(inplace=True)
            combined_report.rename(columns={'index': 'Metric'}, inplace=True)

            all_model_info.append(combined_report)

        detailed_df = pd.concat(all_model_info)
        return detailed_df

    @staticmethod
    def generate_summary_report_dataframe(reports):
        """
        Gera um DataFrame resumido de classificação.
        """
        summary_model_info = []

        for model_name, model_info in reports.items():
            train_report = model_info['train_report'].add_suffix('-train')
            test_report = model_info['test_report'].add_suffix('-test')
            combined_report = train_report.join(test_report)
            
            combined_report['Model'] = model_name
            combined_report.reset_index(inplace=True)
            combined_report.rename(columns={'index': 'Metric'}, inplace=True)

            summary_info = combined_report[combined_report['Metric'].isin(['accuracy', 'macro avg', 'weighted avg', 'balanced_accuracy'])]
            summary_model_info.append(summary_info)

        summary_df = pd.concat(summary_model_info)
        return summary_df
