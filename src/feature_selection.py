import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel

def create_selectors(X_train, y_train):
    selectors = {
        'rfe': create_selector('rfe', n_features_to_select=10),
        'pca': create_selector('pca', n_components=5),
        'rf': create_selector('rf', X_train=X_train, y_train=y_train)
    }
    return selectors

def create_selector(method, X_train=None, y_train=None, n_features_to_select=10, n_components=5):
    if method == 'rfe':
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        return RFE(estimator, n_features_to_select=n_features_to_select)
    elif method == 'pca':
        return PCA(n_components=n_components)
    elif method == 'rf':
        estimator = RandomForestClassifier(n_estimators=100, random_state=0)
        estimator.fit(X_train, y_train)
        return SelectFromModel(estimator)

def get_search_spaces():
    return {
        'rfe': {
            'feature_selection__n_features_to_select': [1, 5, 10, 20, 30, 40, 50]
        },
        'pca': {
            'feature_selection__n_components': [1, 5, 10, 20, 30, 40, 50]
        },
        'rf': {
            'feature_selection__threshold': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
    }


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