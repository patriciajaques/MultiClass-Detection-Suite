import pandas as pd
from core.reporting.classification_model_metrics import ClassificationModelMetrics


class ReportFormatter:
    @staticmethod
    def generate_text_report(model_metrics: ClassificationModelMetrics) -> str:
        """
        Gera relatório textual a partir de um objeto ClassificationModelMetrics
        """
        stage_name = model_metrics.stage_name
        report_output = f"\nEvaluating {stage_name} using {model_metrics.training_type}:\n"
        report_output += f"Hyperparameters: {model_metrics.hyperparameters}\n"

        # Feature information
        if model_metrics.feature_info:
            report_output += ReportFormatter._format_feature_info(
                model_metrics.feature_info)

        # Cross-validation results
        report_output += f"\nCross-Validation Results:\n"
        report_output += f"Average Score: {model_metrics.cv_score:.4f}\n"

        # Train and test results
        report_output += ReportFormatter._format_set_report(
            "Train", model_metrics)
        report_output += ReportFormatter._format_set_report(
            "Test", model_metrics)

        return report_output

    @staticmethod
    def _format_feature_info(feature_info: dict) -> str:
        output = "\nFeature Selection Information:\n"

        if feature_info['type'] == 'pca':
            output += f"Type: PCA\n"
            if 'explained_variance_ratio' in feature_info:
                total_variance = sum(feature_info['explained_variance_ratio'])
                output += f"Total explained variance: {total_variance:.2%}\n"
                output += f"New features: {', '.join(feature_info['new_features'])}\n"
        else:
            output += f"Type: {feature_info['type']}\n"
            output += f"Number of features: {feature_info['n_features']}\n"

        output += f"\n{feature_info['description']}\n"
        return output

    @staticmethod
    def _format_set_report(set_name: str, model_metrics: ClassificationModelMetrics) -> str:
        """Formata o relatório para um conjunto específico (treino ou teste)"""
        output = f"\n{set_name} set class report:\n"

        # Seleciona o relatório correto baseado no conjunto
        class_report = (model_metrics.class_report_train
                        if set_name == "Train"
                        else model_metrics.class_report_test)

        metrics = (model_metrics.train_metrics
                   if set_name == "Train"
                   else model_metrics.test_metrics)

        output += ReportFormatter._dict_to_df(class_report).to_string(index=False)
        output += f"\n\n{set_name} set average metrics:\n"
        output += ReportFormatter._format_metrics(metrics)

        return output


    @staticmethod
    def generate_class_report_dataframe(model_metrics: ClassificationModelMetrics) -> pd.DataFrame:

        # Criar DataFrames separados para treino e teste
        train_df = ReportFormatter._dict_to_df(model_metrics.class_report_train, '-train')
        test_df = ReportFormatter._dict_to_df(model_metrics.class_report_test, '-test')

        # Combinar os DataFrames
        combined_df = train_df.join(test_df, how='outer')

        return combined_df.reset_index().rename(columns={'index': 'Class'})

    @staticmethod
    def generate_avg_metrics_report_dataframe(model_metrics: ClassificationModelMetrics) -> pd.DataFrame:
        """Gera DataFrame com métricas médias do modelo"""
    
        # Criar DataFrames separados para treino e teste
        train_df = pd.DataFrame.from_dict(
            model_metrics.train_metrics, orient='index', columns=['Train'])
        test_df = pd.DataFrame.from_dict(
            model_metrics.test_metrics, orient='index', columns=['Test'])
    
        # Combinar os DataFrames
        combined_df = train_df.join(test_df, how='outer')
    
        # Adicionar linha 'CV Score' na coluna 'Train'
        cv_score_df = pd.DataFrame({'Train': [model_metrics.cv_score]}, index=['CV Score'])
        combined_df = pd.concat([combined_df, cv_score_df])
    
        return combined_df.reset_index().rename(columns={'index': 'Metric'})

    @staticmethod
    def generate_confusion_matrix_report(model_metrics: ClassificationModelMetrics) -> str:
        """Gera relatório formatado das matrizes de confusão"""
        report = f"\nModel: {model_metrics.stage_name}\n"

        # Extrai os labels das classes
        labels = model_metrics.class_labels if model_metrics.class_labels else None

        # Processa matriz de treino
        report += "\nTrain Confusion Matrix:\n"
        report += "Predicted →\nActual ↓\n"
        # Extrai matriz da tupla
        cm_train = model_metrics.confusion_matrix_train[0]
        cm_train_df = pd.DataFrame(
            cm_train,
            index=labels,
            columns=labels
        )
        report += cm_train_df.to_string()

        # Processa matriz de teste
        report += "\n\nTest Confusion Matrix:\n"
        report += "Predicted →\nActual ↓\n"
        cm_test = model_metrics.confusion_matrix_test[0]  # Extrai matriz da tupla
        cm_test_df = pd.DataFrame(
            cm_test,
            index=labels,
            columns=labels
        )
        report += cm_test_df.to_string()

        report += "\n" + "="*50 + "\n"
        return report

    def _format_classification_report(report_dict: dict) -> str:
        """Formata um relatório de classificação a partir do dicionário."""
        output = ""

        # Processar métricas gerais primeiro
        for metric in ['accuracy', 'balanced_accuracy']:
            if metric in report_dict:
                output += f"{metric.capitalize()}: {report_dict[metric]:.4f}\n"

        # Processar métricas por classe
        for class_name, metrics in report_dict.items():
            if isinstance(metrics, dict):  # apenas entradas que são dicionários (métricas por classe)
                metrics_list = []
                if 'precision' in metrics:
                    metrics_list.append(f"Precision: {metrics['precision']:.4f}")
                if 'recall' in metrics:
                    metrics_list.append(f"Recall: {metrics['recall']:.4f}")
                if 'f1-score' in metrics:
                    metrics_list.append(f"F1-Score: {metrics['f1-score']:.4f}")
                if 'support' in metrics:
                    metrics_list.append(f"Support: {int(metrics['support'])}")

                if metrics_list:
                    output += f"Class {class_name} - {', '.join(metrics_list)}\n"

        return output

    
    @staticmethod
    def _format_metrics(metrics: dict) -> str:
        """Formata métricas gerais"""
        return "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items()) + "\n\n"

    @staticmethod
    def _dict_to_df(report_dict, suffix="") -> pd.DataFrame:
        """Gera DataFrame com métricas detalhadas de um relatório de classificação."""
        if not isinstance(report_dict, dict):
            return pd.DataFrame()  # Retorna DataFrame vazio se não for dicionário
    
        # Filtra apenas as entradas que são dicionários (métricas por classe)
        class_metrics = {k: v for k, v in report_dict.items() if isinstance(v, dict)}
    
        df = pd.DataFrame.from_dict(class_metrics, orient='index')
        if suffix:
            df = df.add_suffix(suffix)
        return df
