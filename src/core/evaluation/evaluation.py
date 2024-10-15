import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix
)
from core.preprocessors.feature_selection import FeatureSelection

class Evaluation:
    @staticmethod
    def evaluate_all_models(trained_models, X_train, y_train, X_test, y_test, feature_names=None):
        class_metrics_results = {}
        avg_metrics_results = {}

        for model_name, model_info in trained_models.items():
            model = model_info['model']
            cv_score = model_info['cv_result']

            train_results = Evaluation._evaluate_model(model, X_train, y_train, feature_names)
            test_results = Evaluation._evaluate_model(model, X_test, y_test, feature_names)

            class_metrics_results[model_name] = {
                'train_class_report': train_results['class_report'],
                'train_conf_matrix': train_results['conf_matrix'],
                'test_class_report': test_results['class_report'],
                'test_conf_matrix': test_results['conf_matrix'],
                'selected_features': test_results['selected_features']
            }

            avg_metrics_results[model_name] = {
                'cv_report': cv_score,
                'train_avg_metrics': train_results['avg_metrics'],
                'test_avg_metrics': test_results['avg_metrics'],
                'training_type': model_info['training_type'],
                'hyperparameters': model_info['hyperparameters']
            }

        return class_metrics_results, avg_metrics_results

    @staticmethod
    def _evaluate_model(model, X, y, feature_names=None):
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)

        class_report = Evaluation._generate_class_report(y, y_pred)
        avg_metrics = Evaluation._generate_avg_metrics(y, y_pred, y_prob)
        conf_matrix = Evaluation._generate_conf_matrix(y, y_pred)
        selected_features = Evaluation._extract_selected_features(model, feature_names)

        return {
            'class_report': class_report,
            'avg_metrics': avg_metrics,
            'conf_matrix': conf_matrix,
            'selected_features': selected_features
        }

    @staticmethod
    def _generate_class_report(y_true, y_pred):
        class_report = classification_report(y_true, y_pred, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose().round(2)
        return class_report_df.drop(['accuracy', 'macro avg', 'weighted avg'])

    @staticmethod
    def _generate_avg_metrics(y_true, y_pred, y_prob):
        avg_metrics = {
            'balanced_accuracy': {'f1-score': balanced_accuracy_score(y_true, y_pred)},
            'kappa': {'f1-score': cohen_kappa_score(y_true, y_pred)},
            'auc_pr_macro': {'f1-score': Evaluation._calculate_auc_pr(y_true, y_prob)}
        }
        avg_metrics_df = pd.DataFrame(avg_metrics).transpose().round(2)

        class_report = classification_report(y_true, y_pred, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose().round(2)
        additional_metrics = class_report_df.loc[['accuracy', 'macro avg', 'weighted avg']]

        return pd.concat([avg_metrics_df, additional_metrics])

    @staticmethod
    def _generate_conf_matrix(y_true, y_pred):
        conf_matrix = confusion_matrix(y_true, y_pred)
        classes = sorted(set(y_true))
        return pd.DataFrame(
            conf_matrix,
            index=[f'Actual {cls}' for cls in classes],
            columns=[f'Predicted {cls}' for cls in classes]
        )

    @staticmethod
    def _extract_selected_features(model, feature_names):
        if feature_names is not None:
            return FeatureSelection.extract_selected_features(model, feature_names)
        return None

    @staticmethod
    def _calculate_auc_pr(y_true, y_prob):
        auc_pr = [
            average_precision_score(y_true == i, y_prob[:, i])
            for i in range(len(np.unique(y_true)))
        ]
        return np.mean(auc_pr)