from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

from scipy.stats import randint, uniform
from hyperopt import hp
import numpy as np


def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=5000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

def get_param_grids():
    return  {
        'Logistic Regression': {
            'classifier__penalty': ['l1', 'l2', 'elasticnet', None],
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__solver': ['lbfgs', 'liblinear', 'saga']
        },
        'Decision Tree': {
            'classifier__max_depth': [None, 10, 20, 30, 40, 50],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        },
        'Random Forest': {
            'classifier__n_estimators': [10, 50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'classifier__n_estimators': [10, 50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 10],
            'classifier__subsample': [0.5, 0.7, 1.0]
        },
        'SVM': {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'classifier__gamma': ['scale', 'auto']
        },
        'KNN': {
            'classifier__n_neighbors': [3, 5, 10, 20],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'XGBoost': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7],
            'classifier__subsample': [0.7, 0.8, 1.0],
            'classifier__colsample_bytree': [0.7, 0.8, 1.0]
        }
    }

def get_param_distributions():
    return {
        'Logistic Regression': {
            'classifier__penalty': ['l1', 'l2', 'elasticnet', 'None'],
            'classifier__C': uniform(0.1, 100),
            'classifier__solver': ['lbfgs', 'liblinear', 'saga']
        },
        'Decision Tree': {
            'classifier__max_depth': randint(1, 50),
            'classifier__min_samples_split': randint(2, 10),
            'classifier__min_samples_leaf': randint(1, 4)
        },
        'Random Forest': {
            'classifier__n_estimators': randint(10, 200),
            'classifier__max_depth': randint(1, 30),
            'classifier__min_samples_split': randint(2, 10),
            'classifier__min_samples_leaf': randint(1, 4)
        },
        'Gradient Boosting': {
            'classifier__n_estimators': randint(10, 200),
            'classifier__learning_rate': uniform(0.01, 0.2),
            'classifier__max_depth': randint(3, 10),
            'classifier__subsample': uniform(0.5, 1.0)
        },
        'SVM': {
            'classifier__C': uniform(0.1, 100),
            'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'classifier__gamma': ['scale', 'auto']
        },
        'KNN': {
            'classifier__n_neighbors': randint(3, 20),
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'XGBoost': {
            'classifier__n_estimators': randint(100, 300),
            'classifier__learning_rate': uniform(0.01, 0.2),
            'classifier__max_depth': randint(3, 7),
            'classifier__subsample': uniform(0.7, 0.3),
            'classifier__colsample_bytree': uniform(0.7, 0.3)
        }
    }


def get_hyperopt_spaces():
    return {
        'Logistic Regression': {
            'classifier__penalty': hp.choice('classifier__penalty', ['l1', 'l2', 'elasticnet', None]),
            'classifier__C': hp.loguniform('classifier__C', np.log(0.1), np.log(100)),
            'classifier__solver': hp.choice('classifier__solver', ['lbfgs', 'liblinear', 'saga'])
        },
        'Decision Tree': {
            'classifier__max_depth': hp.randint('classifier__max_depth', 1, 50),
            'classifier__min_samples_split': hp.randint('classifier__min_samples_split', 2, 10),
            'classifier__min_samples_leaf': hp.randint('classifier__min_samples_leaf', 1, 4)
        },
        'Random Forest': {
            'classifier__n_estimators': hp.randint('classifier__n_estimators', 10, 200),
            'classifier__max_depth': hp.randint('classifier__max_depth', 1, 30),
            'classifier__min_samples_split': hp.randint('classifier__min_samples_split', 2, 10),
            'classifier__min_samples_leaf': hp.randint('classifier__min_samples_leaf', 1, 4)
        },
        'Gradient Boosting': {
            'classifier__n_estimators': hp.randint('classifier__n_estimators', 10, 200),
            'classifier__learning_rate': hp.uniform('classifier__learning_rate', 0.01, 0.2),
            'classifier__max_depth': hp.randint('classifier__max_depth', 3, 10),
            'classifier__subsample': hp.uniform('classifier__subsample', 0.5, 1.0)
        },
        'SVM': {
            'classifier__C': hp.loguniform('classifier__C', np.log(0.1), np.log(100)),
            'classifier__kernel': hp.choice('classifier__kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'classifier__gamma': hp.choice('classifier__gamma', ['scale', 'auto'])
        },
        'KNN': {
            'classifier__n_neighbors': hp.randint('classifier__n_neighbors', 3, 20),
            'classifier__weights': hp.choice('classifier__weights', ['uniform', 'distance']),
            'classifier__metric': hp.choice('classifier__metric', ['euclidean', 'manhattan', 'minkowski'])
        },
        'XGBoost': {
            'classifier__n_estimators': hp.randint('classifier__n_estimators', 100, 300),
            'classifier__learning_rate': hp.uniform('classifier__learning_rate', 0.01, 0.2),
            'classifier__max_depth': hp.randint('classifier__max_depth', 3, 7),
            'classifier__subsample': hp.uniform('classifier__subsample', 0.7, 1.0),
            'classifier__colsample_bytree': hp.uniform('classifier__colsample_bytree', 0.7, 1.0)
        }
    }

