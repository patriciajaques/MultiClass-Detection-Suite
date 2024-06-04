from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

from scipy.stats import randint
from hyperopt import hp
import numpy as np


def get_models():
    return {
        # 'Logistic Regression': LogisticRegression(max_iter=5000),
        #'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        # 'SVM': SVC(),
        # 'KNN': KNeighborsClassifier(),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

def get_param_grids():
    return  {
        'Logistic Regression': {
            'classifier__penalty': ['l1', 'l2', 'elasticnet', 'None'],
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
            'classifier__C': randint(0.1, 100),
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
            'classifier__learning_rate': hp.uniform('classifier__learning_rate', 0.01, 0.2),
            'classifier__max_depth': randint(3, 10),
            'classifier__subsample': hp.uniform('classifier__subsample', 0.5, 1.0)
        },
        'SVM': {
            'classifier__C': randint(0.1, 100),
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
            'classifier__learning_rate': hp.uniform('classifier__learning_rate', 0.01, 0.2),
            'classifier__max_depth': randint(3, 7),
            'classifier__subsample': hp.uniform('classifier__subsample', 0.7, 1.0),
            'classifier__colsample_bytree': hp.uniform('classifier__colsample_bytree', 0.7, 1.0)
        }
    }

def get_hyperopt_spaces():
    return {
        'Logistic Regression': {
            'penalty': hp.choice('penalty', ['l1', 'l2', 'elasticnet', 'None']),
            'C': hp.loguniform('C', np.log(0.1), np.log(100)),
            'solver': hp.choice('solver', ['lbfgs', 'liblinear', 'saga'])
        },
        'Decision Tree': {
            'max_depth': hp.randint('max_depth', 1, 50),
            'min_samples_split': hp.randint('min_samples_split', 2, 10),
            'min_samples_leaf': hp.randint('min_samples_leaf', 1, 4)
        },
        'Random Forest': {
            'n_estimators': hp.randint('n_estimators', 10, 200),
            'max_depth': hp.randint('max_depth', 1, 30),
            'min_samples_split': hp.randint('min_samples_split', 2, 10),
            'min_samples_leaf': hp.randint('min_samples_leaf', 1, 4)
        },
        'Gradient Boosting': {
            'n_estimators': hp.randint('n_estimators', 10, 200),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'max_depth': hp.randint('max_depth', 3, 10),
            'subsample': hp.uniform('subsample', 0.5, 1.0)
        },
        'SVM': {
            'C': hp.loguniform('C', np.log(0.1), np.log(100)),
            'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'gamma': hp.choice('gamma', ['scale', 'auto'])
        },
        'KNN': {
            'n_neighbors': hp.randint('n_neighbors', 3, 20),
            'weights': hp.choice('weights', ['uniform', 'distance']),
            'metric': hp.choice('metric', ['euclidean', 'manhattan', 'minkowski'])
        },
        'XGBoost': {
            'n_estimators': hp.randint('n_estimators', 100, 300),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'max_depth': hp.randint('max_depth', 3, 7),
            'subsample': hp.uniform('subsample', 0.7, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0)
        }
    }
