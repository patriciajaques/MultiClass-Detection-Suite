import pandas as pd

class ReportFormatter:

    @staticmethod
    def generate_text_report_from_dict(class_metrics_reports, avg_metrics_reports):
        """
        Gera texto para relatórios detalhados de avaliação, incluindo matrizes de confusão.

        Args:
            class_metrics_reports: Dicionário com os relatórios de avaliação por classe.
            avg_metrics_reports: Dicionário com os relatórios de métricas médias.

        Returns:
            report_output: String com o conteúdo dos relatórios.
        """
        report_output = ""

        for model_name, model_info in class_metrics_reports.items():
            avg_info = avg_metrics_reports[model_name]
            report_output += (f"\nEvaluating {model_name} with {avg_info['training_type']}:\n"
                              f"Hyperparameters: {avg_info['hyperparameters']}\n")
            
            selected_features = model_info.get('selected_features', None)
            if selected_features is not None:
                report_output += f"\nSelected Features: {', '.join(str(feature) for feature in selected_features)}\n"
            
            # Relatório do conjunto de treino
            report_output += "\nTrain set class report:\n"
            report_output += ReportFormatter.format_report(model_info['train_class_report'])
            report_output += "\nTrain set average metrics:\n"
            report_output += ReportFormatter.format_report(avg_info['train_avg_metrics'])
            report_output += "\nTrain set confusion matrix:\n"
            report_output += model_info['train_conf_matrix'].to_string() + "\n"
            
            # Relatório do conjunto de teste
            report_output += "\nTest set class report:\n"
            report_output += ReportFormatter.format_report(model_info['test_class_report'])
            report_output += "\nTest set average metrics:\n"
            report_output += ReportFormatter.format_report(avg_info['test_avg_metrics'])
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
    def generate_class_report_dataframe(class_metrics_reports):
        """
        Gera um DataFrame detalhado de classificação por classe.
        
        Args:
            class_metrics_reports: Dicionário com os relatórios de avaliação por classe.

        Returns:
            detailed_df: DataFrame contendo o relatório detalhado de classificação por classe.
        """
        all_class_info = []

        for model_name, model_info in class_metrics_reports.items():
            train_class_report = model_info['train_class_report'].add_suffix('-train')
            test_class_report = model_info['test_class_report'].add_suffix('-test')
            combined_report = train_class_report.join(test_class_report)
            
            combined_report['Model'] = model_name
            combined_report.reset_index(inplace=True)
            combined_report.rename(columns={'index': 'Metric'}, inplace=True)

            all_class_info.append(combined_report)

        detailed_df = pd.concat(all_class_info)
        return detailed_df

    @staticmethod
    def generate_avg_metrics_report_dataframe(avg_metrics_reports):
        """
        Gera um DataFrame resumido de métricas médias.
        
        Args:
            avg_metrics_reports: Dicionário com os relatórios de métricas médias.

        Returns:
            summary_df: DataFrame contendo o relatório resumido de métricas médias.
        """
        summary_model_info = []

        for model_name, model_info in avg_metrics_reports.items():
            train_avg_metrics = model_info['train_avg_metrics'].add_suffix('-train')
            test_avg_metrics = model_info['test_avg_metrics'].add_suffix('-test')
            combined_report = train_avg_metrics.join(test_avg_metrics)
            
            combined_report['Model'] = model_name
            combined_report.reset_index(inplace=True)
            combined_report.rename(columns={'index': 'Metric'}, inplace=True)

            summary_model_info.append(combined_report)

        summary_df = pd.concat(summary_model_info)
        return summary_df
