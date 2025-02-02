import numpy as np
import pandas as pd
from core.reporting.classification_model_metrics import ClassificationModelMetrics


class ReportFormatter:
    DEFAULT_PRECISION = 4

    @staticmethod
    def setup_formatting(precision=4):
        """Configura a formatação global e define a precisão padrão."""
        ReportFormatter.DEFAULT_PRECISION = precision
        pd.set_option('display.float_format', f'{{:.{precision}f}}'.format)
        pd.set_option('display.precision', precision)

    @staticmethod
    def format_float(value):
        """Formata um valor float com a precisão especificada."""
        return f"{value:.{ReportFormatter.DEFAULT_PRECISION}f}"

    @staticmethod
    def _format_dict(metrics: dict) -> str:
        """Formata métricas gerais"""
        formatted_metrics = {k: ReportFormatter.format_float(
            v) for k, v in metrics.items()}
        return "\n".join(f"{k}: {v}" for k, v in formatted_metrics.items())

    @staticmethod
    def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Formata números em um DataFrame."""
        formatted_df = df.copy()
        numeric_cols = formatted_df.select_dtypes(include=['float64']).columns
        for col in numeric_cols:
            formatted_df[col] = formatted_df[col].round(
                ReportFormatter.DEFAULT_PRECISION)
        return formatted_df

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
        report_output += f"Average Score: {ReportFormatter.format_float(model_metrics.cv_score)}\n"

        # Tempo de treinamento
        report_output += f"Training Time: {ReportFormatter.format_float(model_metrics.execution_time)} seconds\n"

        # Train and test results
        report_output += ReportFormatter._format_set_report(
            "Train", model_metrics)
        report_output += ReportFormatter._format_set_report(
            "Validation", model_metrics)
        report_output += ReportFormatter._format_set_report(
            "Test", model_metrics)

        return report_output

    @staticmethod
    def _format_feature_info(feature_info: dict) -> str:
        output = "\nFeature Selection Information:\n"
        if 'original_n_features' in feature_info:
            output += f"Total number of original features: {feature_info['original_n_features']}\n"

        # Primeiro identifica o tipo de processamento de features
        if feature_info['type'] == 'none':
            output += "No feature selection method was applied - using all original features.\n"

        elif feature_info['type'] == 'pca':
            output += f"Type: PCA\n"
            if 'explained_variance_ratio' in feature_info:
                total_variance = sum(feature_info['explained_variance_ratio'])
                output += f"Total explained variance: {total_variance:.2%}\n"
                output += f"New features: {', '.join(feature_info['new_features'])}\n"
                output += f"Number of selected features: {feature_info['n_features']}\n"
        else:
            output += f"Type: {feature_info['type']}\n"
            output += f"Number of selected features: {feature_info['n_features']}\n"

            # Adiciona a lista de features
            if ('features' in feature_info and
                isinstance(feature_info['features'], (list, np.ndarray)) and
                    len(feature_info['features']) > 0):
                output += "\nSelected features:\n"
                # Formata em colunas para melhor legibilidade
                for i, feature in enumerate(feature_info['features'], 1):
                    output += f"{i:3d}. {feature}\n"

        output += f"\n{feature_info['description']}\n"
        return output

    @staticmethod
    def _format_set_report(set_name: str, model_metrics: ClassificationModelMetrics) -> str:
        """Formata o relatório para um conjunto específico (treino, validação ou teste)"""
        output = f"\n{set_name} set class report:\n"

        # Seleciona o relatório correto baseado no conjunto
        if set_name == "Train":
            class_report = model_metrics.class_report_train
            metrics = model_metrics.train_metrics
        elif set_name == "Validation":
            class_report = model_metrics.class_report_val
            metrics = model_metrics.val_metrics
        else:
            class_report = model_metrics.class_report_test
            metrics = model_metrics.test_metrics

        if class_report:
            output += ReportFormatter._dict_to_df(class_report).to_string(index=True)
        if metrics:
            output += f"\n\n{set_name} set average metrics:\n"
            output += ReportFormatter._format_dict(metrics) + "\n"

        return output

    @staticmethod
    def generate_class_report_dataframe(model_metrics: ClassificationModelMetrics) -> pd.DataFrame:

        # Criar DataFrames separados para treino e teste
        train_df = ReportFormatter._dict_to_df(
            model_metrics.class_report_train, '-train')
        val_df = ReportFormatter._dict_to_df(
            model_metrics.class_report_val, '-val')
        test_df = ReportFormatter._dict_to_df(
            model_metrics.class_report_test, '-test')

        # Combinar os DataFrames
        combined_df = train_df.join(
            val_df, how='outer').join(test_df, how='outer')
        return combined_df.reset_index().rename(columns={'index': 'Class'})

    @staticmethod
    def generate_avg_metrics_report_dataframe(model_metrics: ClassificationModelMetrics) -> pd.DataFrame:
        """Gera DataFrame com métricas médias do modelo"""

        # Criar DataFrames separados para treino e teste
        train_df = pd.DataFrame.from_dict(
            model_metrics.train_metrics, orient='index', columns=['Train'])
        if model_metrics.val_metrics:
            val_df = pd.DataFrame.from_dict(
                model_metrics.val_metrics, orient='index', columns=['Validation'])
        test_df = pd.DataFrame.from_dict(
            model_metrics.test_metrics, orient='index', columns=['Test'])

        # Combinar os DataFrames
        if model_metrics.val_metrics:
            combined_df = train_df.join([val_df, test_df], how='outer')
        else:
            combined_df = train_df.join(test_df, how='outer')

        # Adicionar linha 'CV Score' na coluna 'Train'
        cv_score_df = pd.DataFrame(
            {'Train': [model_metrics.cv_score]}, index=['CV Score'])
        combined_df = pd.concat([combined_df, cv_score_df])

        # Adicionar linha 'Execution Time' na coluna 'Train'
        exec_time_df = pd.DataFrame(
            {'Train': [model_metrics.execution_time]}, index=['Execution Time'])
        combined_df = pd.concat([combined_df, exec_time_df])

        combined_df = ReportFormatter.format_dataframe(combined_df.reset_index().rename(
            columns={'index': 'Metric'}))

        return combined_df

    @staticmethod
    def generate_confusion_matrix_report(model_metrics: ClassificationModelMetrics) -> str:
        report = f"\nModel: {model_metrics.stage_name}\n"
        labels = model_metrics.class_labels if model_metrics.class_labels else None

        # Processa matriz de treino
        if model_metrics.confusion_matrix_train is not None:
            report += "\nTrain Confusion Matrix:\n"
            report += "Predicted →\nActual ↓\n"
            cm_train = model_metrics.confusion_matrix_train[0]
            cm_train_df = pd.DataFrame(cm_train, index=labels, columns=labels)
            report += cm_train_df.to_string()

        # Processa matriz de validação
        if model_metrics.confusion_matrix_val is not None:
            report += "\n\nValidation Confusion Matrix:\n"
            report += "Predicted →\nActual ↓\n"
            cm_val = model_metrics.confusion_matrix_val[0]
            cm_val_df = pd.DataFrame(cm_val, index=labels, columns=labels)
            report += cm_val_df.to_string()

        # Processa matriz de teste
        if model_metrics.confusion_matrix_test is not None:
            report += "\n\nTest Confusion Matrix:\n"
            report += "Predicted →\nActual ↓\n"
            cm_test = model_metrics.confusion_matrix_test[0]
            cm_test_df = pd.DataFrame(cm_test, index=labels, columns=labels)
            report += cm_test_df.to_string()

        report += "\n" + "="*50 + "\n"
        return report

    @staticmethod
    def _format_classification_report(report_dict: dict) -> str:
        """Formata um relatório de classificação a partir do dicionário."""
        output = ""

        # Processar métricas gerais primeiro
        for metric in ['accuracy', 'balanced_accuracy']:
            if metric in report_dict:
                output += f"{metric.capitalize()}: {ReportFormatter.format_float(report_dict[metric])}\n"

        # Processar métricas por classe
        for class_name, metrics in report_dict.items():
            # apenas entradas que são dicionários (métricas por classe)
            if isinstance(metrics, dict):
                metrics_list = []
                if 'precision' in metrics:
                    metrics_list.append(
                        f"Precision: {ReportFormatter.format_float(metrics['precision'])}")
                if 'recall' in metrics:
                    metrics_list.append(
                        f"Recall: {ReportFormatter.format_float(metrics['recall'])}")
                if 'f1-score' in metrics:
                    metrics_list.append(
                        f"F1-Score: {ReportFormatter.format_float(metrics['f1-score'])}")
                if 'support' in metrics:
                    metrics_list.append(f"Support: {int(metrics['support'])}")

                if metrics_list:
                    output += f"Class {class_name} - {', '.join(metrics_list)}\n"

        return output

    @staticmethod
    def _dict_to_df(report_dict, suffix="") -> pd.DataFrame:
        """Gera DataFrame com métricas detalhadas de um relatório de classificação."""
        if not isinstance(report_dict, dict):
            return pd.DataFrame()  # Retorna DataFrame vazio se não for dicionário

        # Filtra apenas as entradas que são dicionários (métricas por classe)
        class_metrics = {k: v for k, v in report_dict.items()
                         if isinstance(v, dict)}

        df = pd.DataFrame.from_dict(class_metrics, orient='index')
        df = ReportFormatter.format_dataframe(df)
        if suffix:
            df = df.add_suffix(suffix)
        return df
