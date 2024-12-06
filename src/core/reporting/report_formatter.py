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
        model_parts = model_name.split('_')
        base_model = model_parts[0]
        selector = model_parts[1] if len(model_parts) > 1 else 'none'
        
        report = (f"\nEvaluating {base_model} with {selector} selector using {avg_info['training_type']}:\n"
                  f"Hyperparameters: {avg_info['hyperparameters']}\n")
        
        if 'feature_info' in model_info:
            feature_info = model_info['feature_info']
            
            if feature_info['type'] == 'pca':
                report += "\nPCA Information:\n"
                report += f"Number of components: {feature_info['n_components']}\n"
                if 'explained_variance_ratio' in feature_info:
                    total_variance = sum(feature_info['explained_variance_ratio'])
                    report += f"Total explained variance: {total_variance:.2%}\n"
                    report += "New features: " + ", ".join(feature_info['new_features']) + "\n"
                
            elif feature_info['type'] == 'selector':
                if 'selected_features' in feature_info:
                    report += "\nSelected Features:\n"
                    report += ", ".join(map(str, feature_info['selected_features'])) + "\n"
                    
            report += f"\n{feature_info['description']}\n"

        report += "\nCross-Validation Results:\n"
        report += f"Average Score: {avg_info['cv_report']:.4f}\n"
        
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
        dfs = []
        for model_name, model_info in class_metrics_reports.items():
            train_report = model_info['train_class_report'].add_suffix('-train')
            test_report = model_info['test_class_report'].add_suffix('-test')
            combined_report = train_report.join(test_report)
            combined_report['Model'] = model_name
            
            # Adicionar informações do feature selector
            if 'feature_info' in model_info:
                feature_info = model_info['feature_info']
                combined_report['Feature_Selection_Type'] = feature_info['type']
                combined_report['N_Features'] = feature_info['n_features']
                
                if feature_info['type'] == 'pca' and 'explained_variance_ratio' in feature_info:
                    combined_report['Total_Variance_Explained'] = sum(feature_info['explained_variance_ratio'])
                
            dfs.append(combined_report.reset_index().rename(columns={'index': 'Metric'}))
            
        return pd.concat(dfs, ignore_index=True)

    @staticmethod
    def _prepare_model_report(model_name, model_info):
        train_report = model_info['train_class_report'].add_suffix('-train')
        test_report = model_info['test_class_report'].add_suffix('-test')
        combined_report = train_report.join(test_report)
        combined_report['Model'] = model_name
        return combined_report.reset_index().rename(columns={'index': 'Metric'})

    @staticmethod
    def generate_avg_metrics_report_dataframe(avg_metrics_reports):
        """
        Gera um DataFrame com métricas médias de todos os modelos.
        
        Args:
            avg_metrics_reports (dict): Dicionário com resultados das métricas médias
            
        Returns:
            pd.DataFrame: DataFrame formatado com as métricas médias
        """
        rows = []
        for model_name, model_info in avg_metrics_reports.items():
            row = {'Model': model_name}
            
            # Adicionar score da cross-validation
            row['CV Score'] = model_info['cv_report']
            
            # Adicionar métricas de treino
            train_metrics = model_info['train_avg_metrics']
            row['balanced_accuracy-train'] = train_metrics.loc['balanced_accuracy', 'f1-score']
            row['f1-score-train'] = train_metrics.loc['weighted avg', 'f1-score']
            row['precision-train'] = train_metrics.loc['weighted avg', 'precision']
            row['recall-train'] = train_metrics.loc['weighted avg', 'recall']
            
            # Adicionar métricas de teste
            test_metrics = model_info['test_avg_metrics']
            row['balanced_accuracy-test'] = test_metrics.loc['balanced_accuracy', 'f1-score']
            row['f1-score-test'] = test_metrics.loc['weighted avg', 'f1-score']
            row['precision-test'] = test_metrics.loc['weighted avg', 'precision']
            row['recall-test'] = test_metrics.loc['weighted avg', 'recall']
            
            rows.append(row)
        
        # Criar DataFrame final
        result_df = pd.DataFrame(rows)
        
        # Definir ordem das colunas
        column_order = [
            'Model', 
            'CV Score',
            'balanced_accuracy-train',
            'f1-score-train',
            'precision-train',
            'recall-train',
            'balanced_accuracy-test',
            'f1-score-test',
            'precision-test',
            'recall-test'
        ]
        
        return result_df[column_order]


    @staticmethod
    def _prepare_avg_metrics_report(model_name, model_info):
        train_metrics = model_info['train_avg_metrics'].add_suffix('-train')
        test_metrics = model_info['test_avg_metrics'].add_suffix('-test')
        combined_report = train_metrics.join(test_metrics)
        combined_report['Model'] = model_name
        return combined_report.reset_index().rename(columns={'index': 'Metric'})