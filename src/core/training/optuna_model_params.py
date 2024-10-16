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
        model_methods = {
            'Logistic Regression': OptunaModelParams._suggest_logistic_regression,
            'Decision Tree': OptunaModelParams._suggest_decision_tree,
            'Random Forest': OptunaModelParams._suggest_random_forest,
            'Gradient Boosting': OptunaModelParams._suggest_gradient_boosting,
            'SVM': OptunaModelParams._suggest_svm,
            'KNN': OptunaModelParams._suggest_knn,
            'XGBoost': OptunaModelParams._suggest_xgboost
        }
        
        if model_name not in model_methods:
            raise ValueError(f"Modelo desconhecido: {model_name}")
        
        return model_methods[model_name](trial)

    @staticmethod
    def _suggest_logistic_regression(trial):
        penalty = trial.suggest_categorical('classifier__penalty', ['l1', 'l2'])
        C = trial.suggest_float('classifier__C', 0.01, 10, log=True)
        
        if penalty == 'l1':
            solver = trial.suggest_categorical('classifier__solver', ['liblinear', 'saga'])
        else:  # l2
            solver = trial.suggest_categorical('classifier__solver', ['lbfgs', 'newton-cg', 'sag', 'saga'])
        
        max_iter = trial.suggest_int('classifier__max_iter', 1000, 10000)
        
        return {
            'classifier__penalty': penalty,
            'classifier__C': C,
            'classifier__solver': solver,
            'classifier__max_iter': max_iter
        }

    @staticmethod
    def _suggest_decision_tree(trial):
        return {
            'classifier__max_depth': trial.suggest_categorical('classifier__max_depth', [None, 3, 5, 10, 20, 30]),
            'classifier__min_samples_split': trial.suggest_int('classifier__min_samples_split', 2, 20),
            'classifier__min_samples_leaf': trial.suggest_int('classifier__min_samples_leaf', 1, 10)
        }

    @staticmethod
    def _suggest_random_forest(trial):
        return {
            'classifier__n_estimators': trial.suggest_int('classifier__n_estimators', 50, 300, step=50),
            'classifier__max_depth': trial.suggest_int('classifier__max_depth', 3, 30),
            'classifier__min_samples_split': trial.suggest_int('classifier__min_samples_split', 2, 20),
            'classifier__min_samples_leaf': trial.suggest_int('classifier__min_samples_leaf', 1, 10),
            'classifier__max_features': trial.suggest_categorical('classifier__max_features', ['sqrt', 'log2', None])
        }

    @staticmethod
    def _suggest_gradient_boosting(trial):
        return {
            'classifier__n_estimators': trial.suggest_int('classifier__n_estimators', 50, 300, step=50),
            'classifier__learning_rate': trial.suggest_float('classifier__learning_rate', 0.01, 0.2),
            'classifier__max_depth': trial.suggest_int('classifier__max_depth', 3, 10),
            'classifier__subsample': trial.suggest_float('classifier__subsample', 0.5, 1.0),
            'classifier__min_samples_split': trial.suggest_int('classifier__min_samples_split', 2, 20),
            'classifier__min_samples_leaf': trial.suggest_int('classifier__min_samples_leaf', 1, 10)
        }

    @staticmethod
    def _suggest_svm(trial):
        params = {
            'classifier__C': trial.suggest_float('classifier__C', 0.01, 10, log=True),
            'classifier__kernel': trial.suggest_categorical('classifier__kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
        }
        if params['classifier__kernel'] in ['rbf', 'poly', 'sigmoid']:
            params['classifier__gamma'] = trial.suggest_float('classifier__gamma', 1e-4, 1e-1, log=True)
        if params['classifier__kernel'] == 'poly':
            params['classifier__degree'] = trial.suggest_int('classifier__degree', 2, 5)
        return params

    @staticmethod
    def _suggest_knn(trial):
        return {
            'classifier__n_neighbors': trial.suggest_int('classifier__n_neighbors', 3, 20),
            'classifier__weights': trial.suggest_categorical('classifier__weights', ['uniform', 'distance']),
            'classifier__metric': trial.suggest_categorical('classifier__metric', ['euclidean', 'manhattan', 'minkowski'])
        }

    @staticmethod
    def _suggest_xgboost(trial):
        return {
            'classifier__n_estimators': trial.suggest_int('classifier__n_estimators', 50, 300, step=50),
            'classifier__learning_rate': trial.suggest_float('classifier__learning_rate', 0.01, 0.2),
            'classifier__max_depth': trial.suggest_int('classifier__max_depth', 3, 10),
            'classifier__subsample': trial.suggest_float('classifier__subsample', 0.7, 1.0),
            'classifier__colsample_bytree': trial.suggest_float('classifier__colsample_bytree', 0.7, 1.0),
            'classifier__reg_alpha': trial.suggest_float('classifier__reg_alpha', 0.0, 1.0),
            'classifier__reg_lambda': trial.suggest_float('classifier__reg_lambda', 0.0, 1.0)
        }