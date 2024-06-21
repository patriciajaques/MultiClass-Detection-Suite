from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

from scipy.stats import randint, uniform
from skopt.space import Real, Integer, Categorical

from training_constants import CROSS_VALIDATION, GRID_SEARCH, RANDOM_SEARCH, BAYESIAN_OPTIMIZATION


def get_models():
    from training import execute_cv, execute_grid_search, execute_random_search, execute_bayesian_optimization

    return {
        #'Logistic Regression': LogisticRegression(max_iter=5000),
        'Decision Tree': DecisionTreeClassifier(),
        #'Random Forest': RandomForestClassifier(),
        # 'Gradient Boosting': GradientBoostingClassifier(),
        # 'SVM': SVC(),
        # 'KNN': KNeighborsClassifier(),
        # 'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

def get_param_grids():
    return {
        'Logistic Regression': [
            {
                'classifier__penalty': ['l2', None],
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
                'classifier__penalty': ['l2', None],
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
        ],
        'Decision Tree': {
            'classifier__max_depth': Categorical([None, 3, 5, 10, 20, 30]),
            'classifier__min_samples_split': Integer(2, 20),
            'classifier__min_samples_leaf': Integer(1, 10)
        },
        'Random Forest': {
            'classifier__n_estimators': Integer(50, 300),
            'classifier__max_depth': Integer(3, 30),
            'classifier__min_samples_split': Integer(2, 20),
            'classifier__min_samples_leaf': Integer(1, 10),
            'classifier__max_features': Categorical(['sqrt', 'log2', None])
        },
        'Gradient Boosting': {
            'classifier__n_estimators': Integer(50, 300),
            'classifier__learning_rate': Real(0.01, 0.2, prior='uniform'),
            'classifier__max_depth': Integer(3, 10),
            'classifier__subsample': Real(0.5, 1.0, prior='uniform'),
            'classifier__min_samples_split': Integer(2, 20),
            'classifier__min_samples_leaf': Integer(1, 10)
        },
        'SVM': [
            {
                'classifier__C': Real(0.01, 10, prior='log-uniform'),
                'classifier__kernel': Categorical(['rbf']),
                'classifier__gamma': Real(1e-4, 1e-1, prior='log-uniform')
            },
            {
                'classifier__C': Real(0.01, 10, prior='log-uniform'),
                'classifier__kernel': Categorical(['linear']),
                'classifier__gamma': Categorical(['scale'])  # `gamma` não se aplica ao kernel `linear`
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
        ],
        'KNN': {
            'classifier__n_neighbors': Integer(3, 20),
            'classifier__weights': Categorical(['uniform', 'distance']),
            'classifier__metric': Categorical(['euclidean', 'manhattan', 'minkowski'])
        },
        'XGBoost': {
            'classifier__n_estimators': Integer(50, 300),
            'classifier__learning_rate': Real(0.01, 0.2, prior='uniform'),
            'classifier__max_depth': Integer(3, 10),
            'classifier__subsample': Real(0.7, 1.0, prior='uniform'),
            'classifier__colsample_bytree': Real(0.7, 1.0, prior='uniform'),
            'classifier__reg_alpha': Real(0.0, 1.0, prior='uniform'),
            'classifier__reg_lambda': Real(0.0, 1.0, prior='uniform')
        }
    }

def get_training_configs(n_iter=50, cv=5):
    from training import execute_cv, execute_grid_search, execute_random_search, execute_bayesian_optimization

    """
    Retorna as configurações de treinamento para diferentes métodos de otimização.

    Args:
        n_iter: Número de iterações para otimização (se aplicável).
        cv: Número de folds para validação cruzada.

    Returns:
        dict: Configurações de treinamento.
    """
    return {
        CROSS_VALIDATION: {
            "function": execute_cv,
            "param_function": None,  # Não há função de parâmetros para cross validation
            "kwargs": {}
        },
        GRID_SEARCH: {
            "function": execute_grid_search,
            "param_function": get_param_grids,  # Função que retorna os parâmetros para grid search
            "kwargs": {"cv": cv}
        },
        RANDOM_SEARCH: {
            "function": execute_random_search,
            "param_function": get_param_distributions,  # Função que retorna os parâmetros para random search
            "kwargs": {"n_iter": n_iter, "cv": cv}
        },
        BAYESIAN_OPTIMIZATION: {
            "function": execute_bayesian_optimization,
            "param_function": get_bayes_search_spaces,  # Função que retorna os parâmetros para bayesian optimization
            "kwargs": {"n_iter": n_iter, "cv": cv}
        }
    }  