import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel

class FeatureSelection:

    @staticmethod
    def create_selectors(X_train, y_train):
        n_features = X_train.shape[1]
        selectors = {
            'rfe': FeatureSelection.create_selector('rfe', X_train=X_train, y_train=y_train, n_features_to_select=min(10, n_features)),
            'pca': FeatureSelection.create_selector('pca', X_train=X_train, n_components=min(5, n_features)),
            #'rf': create_selector('rf', X_train=X_train, y_train=y_train)
        }
        return selectors

    @staticmethod
    def create_selector(method, X_train=None, y_train=None, n_features_to_select=10, n_components=5):
        n_features = X_train.shape[1] if X_train is not None else None
        
        if method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=min(n_features_to_select, n_features))
        elif method == 'pca':
            selector = PCA(n_components=min(n_components, n_features))
        elif method == 'rf':
            estimator = RandomForestClassifier(n_estimators=100, random_state=0)
            estimator.fit(X_train, y_train)
            selector = SelectFromModel(estimator)
            selector.fit(X_train, y_train)
            
            initial_features = selector.get_support().sum()
            print(f"Inicialmente selecionadas {initial_features} características.")
            
            if initial_features == 0:
                selector.threshold_ = np.percentile(selector.estimator_.feature_importances_, 75)
                print(f"Ajuste do limiar para o percentil 75, novas características selecionadas: {selector.get_support().sum()}")

                if selector.get_support().sum() == 0:
                    selector.threshold_ = np.percentile(selector.estimator_.feature_importances_, 50)
                    print(f"Ajuste do limiar para o percentil 50, novas características selecionadas: {selector.get_support().sum()}")

                if selector.get_support().sum() == 0:
                    selector.threshold_ = np.percentile(selector.estimator_.feature_importances_, 25)
                    print(f"Ajuste do limiar para o percentil 25, novas características selecionadas: {selector.get_support().sum()}")
            
            final_features = selector.get_support().sum()
            print(f"Finalmente selecionadas {final_features} características.")
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return selector

    @staticmethod
    def get_search_spaces():
        return {
            'rfe': {
                'feature_selection__n_features_to_select': [1, 5, 10, 20, 30, 40, 50]
            },
            'pca': {
                'feature_selection__n_components': [1, 5, 10, 20, 30, 40, 50]
            },
            'rf': {
                'feature_selection__threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            }
        }

    @staticmethod
    def extract_selected_features(pipeline, feature_names):
        """
        Extrai as características selecionadas pelo seletor de características no pipeline.
        
        Args:
            pipeline: Pipeline treinado.
            feature_names: Lista de nomes das características originais.

        Returns:
            List: Lista de características selecionadas.
        """
        selector = pipeline.named_steps['feature_selection']
        
        if hasattr(selector, 'get_support'):
            mask = selector.get_support()
            selected_features = np.array(feature_names)[mask]
        elif hasattr(selector, 'transform'):
            # Para métodos como PCA que não suportam diretamente 'get_support'
            selected_features = selector.transform(np.arange(len(feature_names)).reshape(1, -1)).flatten()
        else:
            raise ValueError("O seletor não tem métodos para extrair características.")
        
        return selected_features
