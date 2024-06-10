from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

from scipy.stats import randint, uniform
from skopt.space import Real, Integer, Categorical


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
    return {
        'Logistic Regression': [
            {
                'classifier__penalty': ['l2', 'none'],
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__solver': ['lbfgs', 'saga'],
                'classifier__max_iter': [5000, 10000, 20000]
            },
            {
                'classifier__penalty': ['elasticnet'],
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__solver': ['saga'],
                'classifier__l1_ratio': [0.1, 0.5, 0.9],
                'classifier__max_iter': [5000, 10000, 20000]
            }
        ],
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
        'SVM': [
            {
                'classifier__kernel': ['linear'],
                'classifier__C': [0.1, 1, 10, 100]
            },
            {
                'classifier__kernel': ['rbf', 'poly', 'sigmoid'],
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__gamma': ['scale', 'auto']
            },
            {
                'classifier__kernel': ['poly'],
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__gamma': ['scale', 'auto'],
                'classifier__degree': [2, 3, 4]
            },
            {
                'classifier__kernel': ['poly', 'sigmoid'],
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__gamma': ['scale', 'auto'],
                'classifier__coef0': [0.0, 0.5, 1.0]
            }
        ],
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
        'Logistic Regression': [
            {
                'classifier__penalty': ['l2', 'none'],
                'classifier__C': uniform(0.01, 10),
                'classifier__solver': ['lbfgs', 'saga'],
                'classifier__max_iter': [5000, 10000, 20000]
            },
            {
                'classifier__penalty': ['elasticnet'],
                'classifier__C': uniform(0.01, 10),
                'classifier__solver': ['saga'],
                'classifier__l1_ratio': uniform(0.1, 0.9),
                'classifier__max_iter': [5000, 10000, 20000]
            }
        ],
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
            'classifier__subsample': uniform(0.5, 0.5),
        },
        'SVM': [
            {
                'classifier__kernel': ['linear'],
                'classifier__C': uniform(0.1, 100)
            },
            {
                'classifier__kernel': ['rbf', 'poly', 'sigmoid'],
                'classifier__C': uniform(0.1, 100),
                'classifier__gamma': uniform(1e-4, 1e-1)
            },
            {
                'classifier__kernel': ['poly'],
                'classifier__C': uniform(0.1, 100),
                'classifier__gamma': uniform(1e-4, 1e-1),
                'classifier__degree': randint(2, 4)
            },
            {
                'classifier__kernel': ['poly', 'sigmoid'],
                'classifier__C': uniform(0.1, 100),
                'classifier__gamma': uniform(1e-4, 1e-1),
                'classifier__coef0': uniform(0.0, 1.0)
            }
        ],
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

def get_bayes_search_spaces():
    return {
        'Logistic Regression': [
            {
                'classifier__penalty': Categorical(['l2', 'none']),
                'classifier__C': Real(0.1, 100, prior='log-uniform'),
                'classifier__solver': Categorical(['lbfgs', 'saga']),
                'classifier__max_iter': Integer(5000, 20000)
            },
            {
                'classifier__penalty': Categorical(['elasticnet']),
                'classifier__C': Real(0.1, 100, prior='log-uniform'),
                'classifier__solver': Categorical(['saga']),
                'classifier__l1_ratio': Real(0.1, 0.9, prior='uniform'),
                'classifier__max_iter': Integer(5000, 20000)
            }
        ],
        'Decision Tree': {
            'classifier__max_depth': Integer(1, 50),
            'classifier__min_samples_split': Integer(2, 10),
            'classifier__min_samples_leaf': Integer(1, 4)
        },
        'Random Forest': {
            'classifier__n_estimators': Integer(10, 200),
            'classifier__max_depth': Integer(1, 30),
            'classifier__min_samples_split': Integer(2, 10),
            'classifier__min_samples_leaf': Integer(1, 4)
        },
        'Gradient Boosting': {
            'classifier__n_estimators': Integer(10, 200),
            'classifier__learning_rate': Real(0.01, 0.2, prior='uniform'),
            'classifier__max_depth': Integer(3, 10),
            'classifier__subsample': Real(0.5, 1.0, prior='uniform')
        },
        'SVM': [
            {
                'classifier__kernel': Categorical(['linear']),
                'classifier__C': Real(0.1, 100, prior='log-uniform')
            },
            {
                'classifier__kernel': Categorical(['rbf', 'poly', 'sigmoid']),
                'classifier__C': Real(0.1, 100, prior='log-uniform'),
                'classifier__gamma': Real(1e-4, 1e-1, prior='log-uniform')
            },
            {
                'classifier__kernel': Categorical(['poly']),
                'classifier__C': Real(0.1, 100, prior='log-uniform'),
                'classifier__gamma': Real(1e-4, 1e-1, prior='log-uniform'),
                'classifier__degree': Integer(2, 4)
            },
            {
                'classifier__kernel': Categorical(['poly', 'sigmoid']),
                'classifier__C': Real(0.1, 100, prior='log-uniform'),
                'classifier__gamma': Real(1e-4, 1e-1, prior='log-uniform'),
                'classifier__coef0': Real(0.0, 1.0, prior='uniform')
            }
        ],
        'KNN': {
            'classifier__n_neighbors': Integer(3, 20),
            'classifier__weights': Categorical(['uniform', 'distance']),
            'classifier__metric': Categorical(['euclidean', 'manhattan', 'minkowski'])
        },
        'XGBoost': {
            'classifier__n_estimators': Integer(100, 300),
            'classifier__learning_rate': Real(0.01, 0.2, prior='uniform'),
            'classifier__max_depth': Integer(3, 7),
            'classifier__subsample': Real(0.7, 1.0, prior='uniform'),
            'classifier__colsample_bytree': Real(0.7, 1.0, prior='uniform')
        }
    }
