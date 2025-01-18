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
        
        report = (f"\nEvaluating {base_model} with {selector} selector using {avg_info.get('training_type', 'unknown')}:\n"
                  f"Hyperparameters: {avg_info.get('hyperparameters', {})}\n")
        
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
        report += f"Average Score: {avg_info.get('cv_report', 0.0):.4f}\n"
        
        report += ReportFormatter._format_set_report("Train", model_info, avg_info)
        report += ReportFormatter._format_set_report("Test", model_info, avg_info)
        return report

    @staticmethod
    def _format_set_report(set_name, model_info, avg_info):
        try:
            report = f"\n{set_name} set class report:\n"
            class_report_key = f'{set_name.lower()}_class_report'
            report += ReportFormatter._format_classification_report(model_info[class_report_key])
            
            report += f"\n{set_name} set average metrics:\n"
            avg_key = f'{set_name.lower()}_avg_metrics'
            if avg_key in avg_info:
                report += ReportFormatter._format_classification_report(avg_info[avg_key])
            
            return report
        except Exception as e:
            print(f"Erro ao formatar relatório para {set_name}: {str(e)}")
            return f"\nErro ao gerar relatório para {set_name}\n"

    @staticmethod
    def _format_classification_report(report_df):
        try:
            output = ""
            # Verifica se o DataFrame está em formato de dicionário
            if isinstance(report_df, dict):
                report_df = pd.DataFrame(report_df)

            def format_value(val):
                """Formata um valor baseado em seu tipo."""
                try:
                    if isinstance(val, str):
                        # Tenta converter string para float
                        return f"{float(val):.4f}"
                    elif isinstance(val, (int, float)):
                        return f"{val:.4f}"
                    else:
                        return str(val)
                except ValueError:
                    return str(val)

            for index, row in report_df.iterrows():
                try:
                    # Verifica quais colunas estão disponíveis
                    available_metrics = row.index.tolist()
                    
                    if index in ['accuracy', 'balanced_accuracy']:
                        if 'f1-score' in available_metrics:
                            output += f"{index.capitalize()}: {format_value(row['f1-score'])}\n"
                        else:
                            output += f"{index.capitalize()}: {format_value(row.iloc[0])}\n"
                    else:
                        metrics = []
                        if 'precision' in available_metrics:
                            metrics.append(f"Precision: {format_value(row['precision'])}")
                        if 'recall' in available_metrics:
                            metrics.append(f"Recall: {format_value(row['recall'])}")
                        if 'f1-score' in available_metrics:
                            metrics.append(f"F1-Score: {format_value(row['f1-score'])}")
                        if 'support' in available_metrics:
                            try:
                                support_val = int(float(row['support']))
                                metrics.append(f"Support: {support_val}")
                            except (ValueError, TypeError):
                                metrics.append(f"Support: {row['support']}")
                        
                        if metrics:
                            output += f"Class {index} - {', '.join(metrics)}\n"
                        else:
                            metrics = [f"{k}: {format_value(v)}" for k, v in row.items()]
                            output += f"Class {index} - Values: {', '.join(metrics)}\n"
                
                except Exception as e:
                    print(f"Erro ao processar linha {index}: {str(e)}")
                    continue
                    
            return output
            
        except Exception as e:
            print(f"Erro ao formatar relatório de classificação: {str(e)}")
            return "Erro ao gerar relatório de classificação\n"

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
        try:
            rows = []
            for model_name, model_info in avg_metrics_reports.items():
                row = {'Model': model_name}
                
                # Adicionar score da cross-validation
                row['CV Score'] = model_info.get('cv_report', 0.0)
                
                # Função auxiliar para extrair métricas com segurança
                def get_metric_safely(metrics_df, metric_name, column_name='f1-score'):
                    try:
                        if metric_name in metrics_df.index and column_name in metrics_df.columns:
                            return metrics_df.loc[metric_name, column_name]
                        elif metric_name in metrics_df.index:
                            return metrics_df.loc[metric_name].iloc[0]
                        else:
                            return 0.0
                    except Exception:
                        return 0.0
                
                # Adicionar métricas de treino
                train_metrics = model_info.get('train_avg_metrics', pd.DataFrame())
                if not train_metrics.empty:
                    row['balanced_accuracy-train'] = get_metric_safely(train_metrics, 'balanced_accuracy')
                    row['f1-score-train'] = get_metric_safely(train_metrics, 'weighted avg')
                    row['precision-train'] = get_metric_safely(train_metrics, 'weighted avg', 'precision')
                    row['recall-train'] = get_metric_safely(train_metrics, 'weighted avg', 'recall')
                else:
                    row.update({
                        'balanced_accuracy-train': 0.0,
                        'f1-score-train': 0.0,
                        'precision-train': 0.0,
                        'recall-train': 0.0
                    })
                
                # Adicionar métricas de teste
                test_metrics = model_info.get('test_avg_metrics', pd.DataFrame())
                if not test_metrics.empty:
                    row['balanced_accuracy-test'] = get_metric_safely(test_metrics, 'balanced_accuracy')
                    row['f1-score-test'] = get_metric_safely(test_metrics, 'weighted avg')
                    row['precision-test'] = get_metric_safely(test_metrics, 'weighted avg', 'precision')
                    row['recall-test'] = get_metric_safely(test_metrics, 'weighted avg', 'recall')
                else:
                    row.update({
                        'balanced_accuracy-test': 0.0,
                        'f1-score-test': 0.0,
                        'precision-test': 0.0,
                        'recall-test': 0.0
                    })
                
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
            
            # Garantir que todas as colunas existam, mesmo que vazias
            for col in column_order:
                if col not in result_df.columns:
                    result_df[col] = 0.0
            
            return result_df[column_order]
            
        except Exception as e:
            print(f"Erro ao gerar relatório de métricas médias: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # Retornar um DataFrame vazio com as colunas esperadas
            return pd.DataFrame(columns=[
                'Model', 'CV Score',
                'balanced_accuracy-train', 'f1-score-train', 'precision-train', 'recall-train',
                'balanced_accuracy-test', 'f1-score-test', 'precision-test', 'recall-test'
            ])


    @staticmethod
    def _prepare_avg_metrics_report(model_name, model_info):
        train_metrics = model_info['train_avg_metrics'].add_suffix('-train')
        test_metrics = model_info['test_avg_metrics'].add_suffix('-test')
        combined_report = train_metrics.join(test_metrics)
        combined_report['Model'] = model_name
        return combined_report.reset_index().rename(columns={'index': 'Metric'})