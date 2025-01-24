from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from core.models.multiclass.multiclass_model_params import MulticlassModelParams
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

class BehaviorModelParams(MulticlassModelParams):
    """
    Classe especializada para parâmetros de modelos específicos para a classificação
    de comportamentos de aprendizagem em Sistemas Tutores Inteligentes.
    """

    def _create_base_models(self) -> Dict[str, BaseEstimator]:
        models = super()._create_base_models()
        return models

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
        return [{
            'classifier__hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
            'classifier__activation': ['relu', 'tanh'],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'classifier__batch_size': [32, 64],
            'classifier__learning_rate': ['adaptive'],
            'classifier__max_iter': [2000],
            'classifier__early_stopping': [True],  # Adicionar
            'classifier__validation_fraction': [0.1],  # Adicionar
            'classifier__n_iter_no_change': [10],  # Adicionar
            'classifier__solver': ['adam'],
            'classifier__learning_rate_init': [0.001, 0.01]
        }]


    def _get_xgboost_space(self):
        return {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 6, 9],
            'classifier__learning_rate': [0.01, 0.1, 0.3],
            'classifier__subsample': [0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.8, 0.9, 1.0],
            # Remover parâmetros problemáticos
            # 'classifier__early_stopping_rounds': [10],
            # 'classifier__eval_metric': ['mlogloss']
        }


    def _get_gradient_boosting_space(self):
        return {
            'classifier__n_estimators': [300],  # Aumentar máximo
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth': [3, 5, 7],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__subsample': [0.8, 0.9, 1.0],
            'classifier__validation_fraction': [0.1],  # Adicionar
            'classifier__n_iter_no_change': [10],  # Adicionar
            'classifier__tol': [1e-4]  # Adicionar
        }
