import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, precision_recall_curve

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
            class_metrics_results: Dicionário com os relatórios de avaliação por classe.
            avg_metrics_results: Dicionário com os relatórios de métricas médias.
        """
        class_metrics_results = {}
        avg_metrics_results = {}

        for model_name, model_info in trained_models.items():
            model = model_info['model']

            # Acessar os resultados da validação cruzada do BayesSearchCV
            cv_score = model_info['cv_result']

            # Avaliar o modelo no conjunto de treinamento usando predict
            train_class_report, train_conf_matrix, train_selected_features, train_avg_metrics = Evaluation.evaluate_model(model, X_train, y_train, feature_names)

            # Avaliar o modelo no conjunto de teste
            test_class_report, test_conf_matrix, test_selected_features, test_avg_metrics = Evaluation.evaluate_model(model, X_test, y_test, feature_names)
            
            class_metrics_results[model_name] = {
                'train_class_report': train_class_report,
                'train_conf_matrix': train_conf_matrix,
                'test_class_report': test_class_report,
                'test_conf_matrix': test_conf_matrix,
                'selected_features': test_selected_features
            }
            
            avg_metrics_results[model_name] = {
                'cv_report': cv_score,
                'train_avg_metrics': train_avg_metrics,
                'test_avg_metrics': test_avg_metrics,
                'training_type': model_info['training_type'],
                'hyperparameters': model_info['hyperparameters']
            }

        return class_metrics_results, avg_metrics_results

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
            class_metrics_df: DataFrame contendo o relatório de classificação por classe.
            conf_matrix_df: DataFrame contendo a matriz de confusão.
            selected_features: Lista de características selecionadas (se aplicável).
            avg_metrics_df: DataFrame contendo as métricas médias.
        """
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)  # Supondo que o modelo possui o método predict_proba para obter probabilidades

        # Inicializa os relatórios
        class_report = classification_report(y, y_pred, output_dict=True)
        avg_metrics = {}

        # Adiciona balanced_accuracy ao relatório de métricas médias
        bal_acc = balanced_accuracy_score(y, y_pred)
        avg_metrics['balanced_accuracy'] = {'precision': None, 'recall': None, 'f1-score': bal_acc, 'support': None}
        
        # Cálculo do Kappa e adição ao relatório de métricas médias
        kappa = cohen_kappa_score(y, y_pred)
        avg_metrics['kappa'] = {'precision': None, 'recall': None, 'f1-score': kappa, 'support': None}
        
        # Cálculo do AUC-PR e adição ao relatório de métricas médias
        auc_pr_macro = Evaluation.calculate_auc_pr(y, y_prob)
        avg_metrics['auc_pr_macro'] = {'precision': None, 'recall': None, 'f1-score': auc_pr_macro, 'support': None}

        # Converte os relatórios em DataFrames
        class_report_df = pd.DataFrame(class_report).transpose().round(2)
        avg_metrics_df = pd.DataFrame(avg_metrics).transpose().round(2)

        # Adiciona accuracy, macro avg e weighted avg ao relatório de métricas médias
        avg_metrics_df = pd.concat([avg_metrics_df, class_report_df.loc[['accuracy', 'macro avg', 'weighted avg']]])

        # Remove as métricas médias do relatório por classe
        class_report_df = class_report_df.drop(['accuracy', 'macro avg', 'weighted avg'])

        # Criação da matriz de confusão
        conf_matrix = confusion_matrix(y, y_pred)
        conf_matrix_df = pd.DataFrame(conf_matrix, 
                                      index=[f'Actual {cls}' for cls in sorted(set(y))], 
                                      columns=[f'Predicted {cls}' for cls in sorted(set(y))])

        # Extração das características selecionadas, se aplicável
        if feature_names is not None:
            selected_features = FeatureSelection.extract_selected_features(model, feature_names)
        else:
            selected_features = None

        return class_report_df, conf_matrix_df, selected_features, avg_metrics_df


    @staticmethod
    def calculate_auc_pr(y_true, y_prob):
        """
        # Calcula a AUC-PR macro iterando sobre cada classe, tratando-a como positiva e as demais como negativas, e calculando a média das AUC-PRs.
        Args:
            y_true (array-like): Classes verdadeiras.
            y_prob (array-like): Probabilidades previstas pelo modelo.

        Returns:
            float: AUC-PR média (macro) para todas as classes.
        """
        
        # Cálculo do AUC-PR para cada classe
        auc_pr = []
        for i in range(len(np.unique(y_true))):
            auc_pr.append(average_precision_score(y_true == i, y_prob[:, i]))
        
        auc_pr_macro = np.mean(auc_pr)
        
        return auc_pr_macro


