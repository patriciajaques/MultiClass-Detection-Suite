from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

from skopt.space import Real, Integer, Categorical


def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=5000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'SVM': SVC(probability=True),
        'KNN': KNeighborsClassifier(),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
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

