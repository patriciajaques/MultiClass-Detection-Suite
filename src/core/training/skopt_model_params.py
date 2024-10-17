from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

from skopt.space import Real, Integer, Categorical
from src.core.training.model_params import ModelParams

class SkoptModelParams(ModelParams):

    @staticmethod
    def get_bayes_search_spaces():
        return {
            'Logistic Regression': SkoptModelParams._get_logistic_regression_space(),
            'Decision Tree': SkoptModelParams._get_decision_tree_space(),
            'Random Forest': SkoptModelParams._get_random_forest_space(),
            'Gradient Boosting': SkoptModelParams._get_gradient_boosting_space(),
            'SVM': SkoptModelParams._get_svm_space(),
            'KNN': SkoptModelParams._get_knn_space(),
            'XGBoost': SkoptModelParams._get_xgboost_space()
        }

    @staticmethod
    def _get_logistic_regression_space():
        return [
            {
                'classifier__penalty': Categorical(['l2']),
                'classifier__C': Real(0.01, 10, prior='log-uniform'),
                'classifier__solver': Categorical(['lbfgs', 'saga']),
                'classifier__max_iter': Integer(1000, 10000)
            },
            {
                'classifier__penalty': Categorical(['l1']),
                'classifier__C': Real(0.01, 10, prior='log-uniform'),
                'classifier__solver': Categorical(['liblinear', 'saga']),
                'classifier__max_iter': Integer(1000, 10000)
            }
        ]

    @staticmethod
    def _get_decision_tree_space():
        return {
            'classifier__max_depth': Categorical([None, 3, 5, 10, 20, 30]),
            'classifier__min_samples_split': Integer(2, 20),
            'classifier__min_samples_leaf': Integer(1, 10)
        }

    @staticmethod
    def _get_random_forest_space():
        return {
            'classifier__n_estimators': Integer(50, 300),
            'classifier__max_depth': Integer(3, 30),
            'classifier__min_samples_split': Integer(2, 20),
            'classifier__min_samples_leaf': Integer(1, 10),
            'classifier__max_features': Categorical(['sqrt', 'log2', None])
        }

    @staticmethod
    def _get_gradient_boosting_space():
        return {
            'classifier__n_estimators': Integer(50, 300),
            'classifier__learning_rate': Real(0.01, 0.2, prior='uniform'),
            'classifier__max_depth': Integer(3, 10),
            'classifier__subsample': Real(0.5, 1.0, prior='uniform'),
            'classifier__min_samples_split': Integer(2, 20),
            'classifier__min_samples_leaf': Integer(1, 10)
        }

    @staticmethod
    def _get_svm_space():
        return [
            {
                'classifier__C': Real(0.01, 10, prior='log-uniform'),
                'classifier__kernel': Categorical(['rbf']),
                'classifier__gamma': Real(1e-4, 1e-1, prior='log-uniform')
            },
            {
                'classifier__C': Real(0.01, 10, prior='log-uniform'),
                'classifier__kernel': Categorical(['linear']),
                'classifier__gamma': Categorical(['scale'])
            },
            {
                'classifier__C': Real(0.01, 10, prior='log-uniform'),
                'classifier__kernel': Categorical(['poly']),
                'classifier__gamma': Real(1e-4, 1e-1, prior='log-uniform'),
                'classifier__degree': Integer(2, 5)
            },
            {
                'classifier__C': Real(0.01, 10, prior='log-uniform'),
                'classifier__kernel': Categorical(['sigmoid']),
                'classifier__gamma': Real(1e-4, 1e-1, prior='log-uniform')
            }
        ]

    @staticmethod
    def _get_knn_space():
        return {
            'classifier__n_neighbors': Integer(3, 20),
            'classifier__weights': Categorical(['uniform', 'distance']),
            'classifier__metric': Categorical(['euclidean', 'manhattan', 'minkowski'])
        }

    @staticmethod
    def _get_xgboost_space():
        return {
            'classifier__n_estimators': Integer(50, 300),
            'classifier__learning_rate': Real(0.01, 0.2, prior='uniform'),
            'classifier__max_depth': Integer(3, 10),
            'classifier__subsample': Real(0.7, 1.0, prior='uniform'),
            'classifier__colsample_bytree': Real(0.7, 1.0, prior='uniform'),
            'classifier__reg_alpha': Real(0.0, 1.0, prior='uniform'),
            'classifier__reg_lambda': Real(0.0, 1.0, prior='uniform')
        }