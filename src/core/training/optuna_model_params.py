from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

class OptunaModelParams:
    @staticmethod
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

    @staticmethod
    def suggest_hyperparameters(trial, model_name):
        """
        Sugere hiperparâmetros para o modelo especificado usando um trial do Optuna.
        """
        if model_name == 'Logistic Regression':
            penalty = trial.suggest_categorical('classifier__penalty', ['l1', 'l2'])
            C = trial.suggest_loguniform('classifier__C', 0.01, 10)
            solver = trial.suggest_categorical('classifier__solver', ['lbfgs', 'saga'] if penalty == 'l2' else ['liblinear', 'saga'])
            max_iter = trial.suggest_int('classifier__max_iter', 1000, 10000)
            return {
                'classifier__penalty': penalty,
                'classifier__C': C,
                'classifier__solver': solver,
                'classifier__max_iter': max_iter
            }
        
        elif model_name == 'Decision Tree':
            max_depth = trial.suggest_categorical('classifier__max_depth', [None, 3, 5, 10, 20, 30])
            min_samples_split = trial.suggest_int('classifier__min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('classifier__min_samples_leaf', 1, 10)
            return {
                'classifier__max_depth': max_depth,
                'classifier__min_samples_split': min_samples_split,
                'classifier__min_samples_leaf': min_samples_leaf
            }
        
        elif model_name == 'Random Forest':
            n_estimators = trial.suggest_int('classifier__n_estimators', 50, 300, step=50)
            max_depth = trial.suggest_int('classifier__max_depth', 3, 30)
            min_samples_split = trial.suggest_int('classifier__min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('classifier__min_samples_leaf', 1, 10)
            max_features = trial.suggest_categorical('classifier__max_features', ['sqrt', 'log2', None])
            return {
                'classifier__n_estimators': n_estimators,
                'classifier__max_depth': max_depth,
                'classifier__min_samples_split': min_samples_split,
                'classifier__min_samples_leaf': min_samples_leaf,
                'classifier__max_features': max_features
            }
        
        elif model_name == 'Gradient Boosting':
            n_estimators = trial.suggest_int('classifier__n_estimators', 50, 300, step=50)
            learning_rate = trial.suggest_uniform('classifier__learning_rate', 0.01, 0.2)
            max_depth = trial.suggest_int('classifier__max_depth', 3, 10)
            subsample = trial.suggest_uniform('classifier__subsample', 0.5, 1.0)
            min_samples_split = trial.suggest_int('classifier__min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('classifier__min_samples_leaf', 1, 10)
            return {
                'classifier__n_estimators': n_estimators,
                'classifier__learning_rate': learning_rate,
                'classifier__max_depth': max_depth,
                'classifier__subsample': subsample,
                'classifier__min_samples_split': min_samples_split,
                'classifier__min_samples_leaf': min_samples_leaf
            }
        
        elif model_name == 'SVM':
            C = trial.suggest_loguniform('classifier__C', 0.01, 10)
            kernel = trial.suggest_categorical('classifier__kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
            if kernel == 'rbf':
                gamma = trial.suggest_loguniform('classifier__gamma', 1e-4, 1e-1)
                return {
                    'classifier__C': C,
                    'classifier__kernel': kernel,
                    'classifier__gamma': gamma
                }
            elif kernel == 'linear':
                return {
                    'classifier__C': C,
                    'classifier__kernel': kernel,
                    'classifier__gamma': 'scale'
                }
            elif kernel == 'poly':
                gamma = trial.suggest_loguniform('classifier__gamma', 1e-4, 1e-1)
                degree = trial.suggest_int('classifier__degree', 2, 5)
                return {
                    'classifier__C': C,
                    'classifier__kernel': kernel,
                    'classifier__gamma': gamma,
                    'classifier__degree': degree
                }
            elif kernel == 'sigmoid':
                gamma = trial.suggest_loguniform('classifier__gamma', 1e-4, 1e-1)
                return {
                    'classifier__C': C,
                    'classifier__kernel': kernel,
                    'classifier__gamma': gamma
                }
        
        elif model_name == 'KNN':
            n_neighbors = trial.suggest_int('classifier__n_neighbors', 3, 20)
            weights = trial.suggest_categorical('classifier__weights', ['uniform', 'distance'])
            metric = trial.suggest_categorical('classifier__metric', ['euclidean', 'manhattan', 'minkowski'])
            return {
                'classifier__n_neighbors': n_neighbors,
                'classifier__weights': weights,
                'classifier__metric': metric
            }
        
        elif model_name == 'XGBoost':
            n_estimators = trial.suggest_int('classifier__n_estimators', 50, 300, step=50)
            learning_rate = trial.suggest_uniform('classifier__learning_rate', 0.01, 0.2)
            max_depth = trial.suggest_int('classifier__max_depth', 3, 10)
            subsample = trial.suggest_uniform('classifier__subsample', 0.7, 1.0)
            colsample_bytree = trial.suggest_uniform('classifier__colsample_bytree', 0.7, 1.0)
            reg_alpha = trial.suggest_uniform('classifier__reg_alpha', 0.0, 1.0)
            reg_lambda = trial.suggest_uniform('classifier__reg_lambda', 0.0, 1.0)
            return {
                'classifier__n_estimators': n_estimators,
                'classifier__learning_rate': learning_rate,
                'classifier__max_depth': max_depth,
                'classifier__subsample': subsample,
                'classifier__colsample_bytree': colsample_bytree,
                'classifier__reg_alpha': reg_alpha,
                'classifier__reg_lambda': reg_lambda
            }
        
        else:
            raise ValueError(f"Modelo desconhecido: {model_name}")
