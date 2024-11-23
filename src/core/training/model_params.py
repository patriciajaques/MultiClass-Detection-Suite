from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

class ModelParams:


    @staticmethod
    def get_available_models():
        return list(ModelParams.get_models().keys())

    @staticmethod
    def get_models():
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

    @staticmethod
    def get_param_space(model_name):
        """
        Retorna o espaço de parâmetros para um modelo específico.
        """
        param_spaces = {
            'Logistic Regression': ModelParams._get_logistic_regression_params(),
            'Decision Tree': ModelParams._get_decision_tree_params(),
            'Random Forest': ModelParams._get_random_forest_params(),
            'Gradient Boosting': ModelParams._get_gradient_boosting_space(),
            'SVM': ModelParams._get_svm_space(),
            'KNN': ModelParams._get_knn_space(),
            'XGBoost': ModelParams._get_xgboost_space(),
            'Naive Bayes': ModelParams._get_naive_bayes_space(),
            'MLP': ModelParams._get_mlp_space()
        }
        return param_spaces.get(model_name, {})

    @staticmethod
    def _get_logistic_regression_params(): # para MNIST
        """
        Define o espaço de hiperparâmetros para Regressão Logística otimizada para o problema MNIST.
        
        Returns:
            dict: Dicionário com os espaços de busca dos hiperparâmetros
        
        Justificativa dos parâmetros:
        - penalty: 'l2' é mais adequado para problemas multiclasse e evita overfitting
        - solver: 
            - 'lbfgs': eficiente para multiclasse e datasets médios
            - 'sag': bom para datasets grandes
            - 'newton-cg': preciso para problemas multiclasse
        - C: valores mais granulares entre 0.1 e 10 para encontrar o melhor balanço
        - max_iter: valores maiores para garantir convergência
        """
        return {
            'classifier__penalty': ['l2'],  # l2 é mais estável para multiclasse
            'classifier__solver': ['lbfgs', 'newton-cg', 'sag'],  # Solvers eficientes para multiclasse
            'classifier__C': [0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0, 7.0, 10.0],  # Mais granularidade
            'classifier__max_iter': [2000, 3000, 4000],  # Valores maiores para garantir convergência
            'classifier__multi_class': ['multinomial'],  # Melhor para problemas multiclasse verdadeiros
            'classifier__class_weight': ['balanced', None]  # Testar com e sem balanceamento
        }

    @staticmethod
    def _get_decision_tree_params():
        """
        Parâmetros específicos para Árvore de Decisão
        """
        return {
            'classifier__max_depth': [None, 3, 5, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4, 6, 8, 10],
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__splitter': ['best', 'random'],
            'classifier__max_features': ['sqrt', 'log2', None]
        }

    @staticmethod
    def _get_random_forest_params():
        """
        Parâmetros específicos para Random Forest
        """
        return {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 3, 5, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4, 6, 8, 10],
            'classifier__max_features': ['sqrt', 'log2', None],
            'classifier__bootstrap': [True, False]
        }

    @staticmethod
    def _get_gradient_boosting_space():
        return {
            'classifier__n_estimators': [50, 100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'classifier__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'classifier__subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'classifier__min_samples_split': [2, 5, 10, 15, 20],
            'classifier__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }

    @staticmethod
    def _get_svm_space():
        return [
            {
                'classifier__C': [0.01, 0.1, 1, 10],
                'classifier__kernel': ['rbf'],
                'classifier__gamma': [1e-4, 1e-3, 1e-2, 1e-1]
            },
            {
                'classifier__C': [0.01, 0.1, 1, 10],
                'classifier__kernel': ['linear'],
                'classifier__gamma': ['scale']
            },
            {
                'classifier__C': [0.01, 0.1, 1, 10],
                'classifier__kernel': ['poly'],
                'classifier__gamma': [1e-4, 1e-3, 1e-2, 1e-1],
                'classifier__degree': [2, 3, 4, 5]
            },
            {
                'classifier__C': [0.01, 0.1, 1, 10],
                'classifier__kernel': ['sigmoid'],
                'classifier__gamma': [1e-4, 1e-3, 1e-2, 1e-1]
            }
        ]

    @staticmethod
    def _get_knn_space():
        return {
            'classifier__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 20],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
        }

    @staticmethod
    def _get_xgboost_space():
        return {
            'classifier__n_estimators': [50, 100, 150, 200, 250, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'classifier__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'classifier__subsample': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            'classifier__colsample_bytree': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            'classifier__reg_alpha': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            'classifier__reg_lambda': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        }
    
    @staticmethod
    def _get_naive_bayes_space():
        """
        Define o espaço de hiperparâmetros para Gaussian Naive Bayes.
        
        O GaussianNB tem poucos hiperparâmetros para otimizar, mas incluímos
        var_smoothing que pode ajudar com dados numéricos.
        
        Returns:
            dict: Dicionário com os espaços de busca dos hiperparâmetros
        """
        return {
            'classifier__var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        }

    @staticmethod
    def _get_mlp_space():
        """
        Define o espaço de hiperparâmetros para MLP Classifier.
        
        Parâmetros escolhidos considerando:
        - hidden_layer_sizes: várias arquiteturas de rede, de simples a complexas
        - activation: funções de ativação disponíveis
        - solver: algoritmos de otimização apropriados
        - alpha: valores para regularização L2
        - learning_rate: estratégias de taxa de aprendizado
        - batch_size: diferentes tamanhos de batch para treino
        
        Returns:
            dict: Dicionário com os espaços de busca dos hiperparâmetros
        """
        # Primeiro conjunto de parâmetros (comum a todos os solvers)
        base_params = {
            'classifier__hidden_layer_sizes': [
                (50,), (100,),            # Single layer
                (50, 25), (100, 50),      # Two layers
                (100, 50, 25)             # Three layers
            ],
            'classifier__activation': ['relu', 'tanh'],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'classifier__batch_size': [32, 64, 128, 256],
            'classifier__max_iter': [1000, 2000],
        }

        # Parâmetros específicos para diferentes solvers
        solver_specific = [
            # Configuração para 'adam'
            {
                **base_params,
                'classifier__solver': ['adam'],
                'classifier__learning_rate': ['constant', 'adaptive'],
                'classifier__learning_rate_init': [0.001, 0.01],
                'classifier__early_stopping': [True],
                'classifier__validation_fraction': [0.1, 0.2],
                'classifier__n_iter_no_change': [10, 20],
                'classifier__beta_1': [0.9],
                'classifier__beta_2': [0.999]
            },
            # Configuração para 'sgd'
            {
                **base_params,
                'classifier__solver': ['sgd'],
                'classifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
                'classifier__learning_rate_init': [0.001, 0.01],
                'classifier__early_stopping': [True],
                'classifier__validation_fraction': [0.1, 0.2],
                'classifier__n_iter_no_change': [10, 20],
                'classifier__momentum': [0.8, 0.9],
                'classifier__nesterovs_momentum': [True]
            },
            # Configuração para 'lbfgs'
            {
                **base_params,
                'classifier__solver': ['lbfgs'],
                'classifier__max_iter': [1000, 2000, 3000]  # lbfgs pode precisar de mais iterações
            }
        ]

        return solver_specific
