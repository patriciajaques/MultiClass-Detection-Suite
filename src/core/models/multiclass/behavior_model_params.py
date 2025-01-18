from core.models.multiclass.multiclass_model_params import MulticlassModelParams

class BehaviorModelParams(MulticlassModelParams):
    """
    Classe especializada para parâmetros de modelos específicos para a classificação
    de comportamentos de aprendizagem em Sistemas Tutores Inteligentes.
    """
    def _get_logistic_regression_params(self):
        """
        Parâmetros otimizados para Regressão Logística na classificação de comportamentos.
        """
        return {
            'classifier__penalty': ['l2'],
            'classifier__C': [0.01, 0.1, 1.0, 10.0],  # Maior range para regularização
            'classifier__solver': ['lbfgs', 'newton-cg'],  # Solvers mais eficientes para multiclasse
            'classifier__max_iter': [5000],  # Aumentado para garantir convergência
            'classifier__class_weight': ['balanced']  # Importante para classes desbalanceadas
        }

    def _get_random_forest_params(self):
        """
        Parâmetros otimizados para Random Forest na classificação de comportamentos.
        """
        return {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [10, 20, 30, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2'],
            'classifier__class_weight': ['balanced', 'balanced_subsample'],
            'classifier__bootstrap': [True],
            'classifier__criterion': ['gini', 'entropy']  # Ambos critérios podem ser úteis
        }

    def _get_gradient_boosting_space(self):
        """
        Parâmetros otimizados para Gradient Boosting na classificação de comportamentos.
        """
        return {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1],  # Taxas menores para melhor generalização
            'classifier__max_depth': [3, 5, 7],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__subsample': [0.8, 0.9, 1.0]  # Adiciona subamostragem para reduzir overfitting
        }

    def _get_svm_space(self):
        """
        Parâmetros otimizados para SVM na classificação de comportamentos.
        """
        return [
            # {
            #     'classifier__C': [0.1, 1.0, 10.0],
            #     'classifier__kernel': ['rbf'],
            #     'classifier__gamma': ['scale', 0.1],
            #     'classifier__class_weight': ['balanced']
            # },
            {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__kernel': ['linear'],
                'classifier__class_weight': ['balanced']
            }
        ]

    def _get_mlp_space(self):
        """
        Parâmetros otimizados para MLP na classificação de comportamentos.
        """
        return [
            {
                'classifier__hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                'classifier__activation': ['relu', 'tanh'],
                'classifier__alpha': [0.0001, 0.001, 0.01],
                'classifier__batch_size': [32, 64],
                'classifier__learning_rate': ['adaptive'],
                'classifier__max_iter': [2000],
                'classifier__solver': ['adam'],
                'classifier__learning_rate_init': [0.001, 0.01]
            }
        ]

    def _get_xgboost_space(self):
        """
        Parâmetros otimizados para XGBoost na classificação de comportamentos.
        """
        return {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__subsample': [0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.8, 0.9, 1.0],
            'classifier__min_child_weight': [1, 3, 5],
            'classifier__gamma': [0, 0.1, 0.2]
        }