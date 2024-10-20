from core.training.model_params import ModelParams


class GridSearchModelParams(ModelParams):
    @staticmethod
    def get_param_grid(model_name):
        param_grid_methods = {
            'Logistic Regression': GridSearchModelParams._get_logistic_regression_params,
            'Random Forest': GridSearchModelParams._get_random_forest_params,
            'Gradient Boosting': GridSearchModelParams._get_gradient_boosting_params,
            'SVM': GridSearchModelParams._get_svm_params,
            'XGBoost': GridSearchModelParams._get_xgboost_params
        }
        return param_grid_methods.get(model_name, lambda: {})()

    @staticmethod
    def _get_logistic_regression_params():
        return [
            {
                'classifier__penalty': ['l1'],
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'classifier__solver': ['liblinear', 'saga']
            },
            {
                'classifier__penalty': ['l2'],
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'classifier__solver': ['lbfgs', 'newton-cg', 'sag', 'saga']
            },
            {
                'classifier__penalty': ['elasticnet'],
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'classifier__solver': ['saga'],
                'classifier__l1_ratio': [0.25, 0.5, 0.75]
            },
            {
                'classifier__penalty': ['none'],
                'classifier__solver': ['lbfgs', 'newton-cg', 'sag', 'saga']
            }
        ]

    @staticmethod
    def _get_random_forest_params():
        return {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }

    @staticmethod
    def _get_gradient_boosting_params():
        return {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 4, 5]
        }

    @staticmethod
    def _get_svm_params():
        return {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf', 'linear'],
            'classifier__gamma': ['scale', 'auto']
        }

    @staticmethod
    def _get_xgboost_params():
        return {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.3],
            'classifier__max_depth': [3, 4, 5]
        }
    
