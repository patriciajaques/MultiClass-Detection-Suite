"""
Copyright (c) 2025 Patricia Jaques
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License. To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from core.models.base_model_params import BaseModelParams

class MulticlassModelParams(BaseModelParams):
    """
    Classe base para problemas de multiclassificação.
    Implementa a funcionalidade base que pode ser estendida por domínios específicos.
    """

    def __init__(self):
        self._model_registry = self._create_base_models()

    def get_models(self) -> Dict[str, BaseEstimator]:
        return self._model_registry
    
    def _create_base_models(self) -> Dict[str, BaseEstimator]:
        """Cria os modelos base para multiclassificação"""
        return {
            'Logistic Regression': LogisticRegression(max_iter=5000),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'SVM': SVC(probability=True),
            'KNN': KNeighborsClassifier(),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
            'Naive Bayes': GaussianNB(),
            'MLP': MLPClassifier(max_iter=1000)
        }

    def get_param_space(self, model_name: str) -> Dict[str, Any]:
        """
        Retorna o espaço de parâmetros para um modelo específico.
        """
        param_methods = {
            'Logistic Regression': self._get_logistic_regression_params,
            'Decision Tree': self._get_decision_tree_params,
            'Random Forest': self._get_random_forest_params,
            'Gradient Boosting': self._get_gradient_boosting_space,
            'SVM': self._get_svm_space,
            'KNN': self._get_knn_space,
            'XGBoost': self._get_xgboost_space,
            'Naive Bayes': self._get_naive_bayes_space,
            'MLP': self._get_mlp_space
        }
        method = param_methods.get(model_name)
        return method() if method else {}

    def _get_logistic_regression_params(self):
        """
        Define o espaço de hiperparâmetros padrão para Regressão Logística em problemas multiclasse.
        """
        return {
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs', 'newton-cg', 'sag'],
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__max_iter': [3000, 5000, 7000],
            'classifier__class_weight': ['balanced', None]
        }

    def _get_decision_tree_params(self):
        """
        Parâmetros padrão para Árvore de Decisão em problemas multiclasse.
        """
        return {
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2']
        }

    def _get_random_forest_params(self):
        """
        Parâmetros padrão para Random Forest em problemas multiclasse.
        """
        return {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2'],
            'classifier__class_weight': ['balanced', 'balanced_subsample']
        }

    def _get_gradient_boosting_space(self):
        """
        Parâmetros padrão para Gradient Boosting em problemas multiclasse.
        """
        return {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.1, 0.3],
            'classifier__max_depth': [3, 5, 7],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }

    def _get_svm_space(self):
        """
        Parâmetros padrão para SVM em problemas multiclasse.
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

    def _get_knn_space(self):
        """
        Parâmetros padrão para KNN em problemas multiclasse.
        """
        return {
            'classifier__n_neighbors': [3, 5, 7, 9],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan']
        }

    def _get_xgboost_space(self):
        """
        Parâmetros padrão para XGBoost em problemas multiclasse.
        """
        return {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 6, 9],
            'classifier__learning_rate': [0.01, 0.1, 0.3],
            'classifier__subsample': [0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.8, 0.9, 1.0]
        }

    def _get_naive_bayes_space(self):
        """
        Parâmetros padrão para Naive Bayes em problemas multiclasse.
        """
        return {
            'classifier__var_smoothing': [1e-5, 1e-4, 1e-3]  # Valores ajustados para maior estabilidade
        }

    def _get_mlp_space(self):
        """
        Parâmetros padrão para MLP em problemas multiclasse.
        """
        base_params = {
            'classifier__hidden_layer_sizes': [
                (100,), (50, 25), (100, 50)
            ],
            'classifier__activation': ['relu', 'tanh'],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'classifier__batch_size': [32, 64, 128],
            'classifier__learning_rate': ['constant', 'adaptive'],
            'classifier__max_iter': [1000, 2000]
        }

        return [
            {
                **base_params,
                'classifier__solver': ['adam'],
                'classifier__learning_rate_init': [0.001, 0.01]
            },
            {
                **base_params,
                'classifier__solver': ['lbfgs']
            }
        ]