import pandas as pd
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix

from feature_selection import FeatureSelection

class Evaluation:

    @staticmethod
    def evaluate_all_models(trained_models, X_train, y_train, X_test, y_test, feature_names=None):
        """
        Gera relatórios de avaliação para todos os modelos nos conjuntos de treino e teste, utilizando os resultados da
        validação cruzada realizada pelo BayesSearchCV.

        Args:
            trained_models: Dicionário de modelos treinados com BayesSearchCV.
            X_train: Conjunto de características de treino.
            y_train: Conjunto de rótulos de treino.
            X_test: Conjunto de características de teste.
            y_test: Conjunto de rótulos de teste.
            feature_names: Lista de nomes das características (opcional).

        Returns:
            results: Dicionário com os relatórios de avaliação.
        """
        results = {}
        for model_name, model_info in trained_models.items():
            model = model_info['model']

            # Acessar os resultados da validação cruzada do BayesSearchCV
            cv_score = model_info['cv_result']

            # Avaliar o modelo no conjunto de treinamento usando predict
            train_report, train_conf_matrix, train_selected_features = Evaluation.evaluate_model(model, X_train, y_train, feature_names)

            # Avaliar o modelo no conjunto de teste
            test_report, test_conf_matrix, test_selected_features = Evaluation.evaluate_model(model, X_test, y_test, feature_names)
            
            results[model_name] = {
                'cv_report': cv_score,
                'train_report': train_report,
                'train_conf_matrix': train_conf_matrix,
                'test_report': test_report,
                'test_conf_matrix': test_conf_matrix,
                'training_type': model_info['training_type'],
                'hyperparameters': model_info['hyperparameters'],
                'selected_features': test_selected_features
            }
        return results

    @staticmethod
    def evaluate_model(model, X, y, feature_names=None):
        """
        Avalia um modelo e retorna as previsões, as métricas de avaliação e as características selecionadas.

        Args:
            model: Modelo treinado.
            X: Conjunto de características.
            y: Conjunto de rótulos.
            feature_names: Lista de nomes das características (opcional).

        Returns:
            report_df: DataFrame contendo o relatório de classificação.
            conf_matrix_df: DataFrame contendo a matriz de confusão.
            selected_features: Lista de características selecionadas (se aplicável).
        """
        y_pred = model.predict(X)
        report = classification_report(y, y_pred, output_dict=True)
        bal_acc = balanced_accuracy_score(y, y_pred)
        report['balanced_accuracy'] = {'precision': None, 'recall': None, 'f1-score': bal_acc, 'support': None}
        report_df = pd.DataFrame(report).transpose().round(2)
        
        conf_matrix = confusion_matrix(y, y_pred)
        conf_matrix_df = pd.DataFrame(conf_matrix, 
                                    index=[f'Actual {cls}' for cls in sorted(set(y))], 
                                    columns=[f'Predicted {cls}' for cls in sorted(set(y))])

        if feature_names is not None:
            selected_features = FeatureSelection.extract_selected_features(model, feature_names)
        else:
            selected_features = None
        
        return report_df, conf_matrix_df, selected_features