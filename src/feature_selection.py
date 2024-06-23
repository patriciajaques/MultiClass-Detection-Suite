import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

# Define a map for selector methods
SELECTOR_MAP = {
    'rfe': RFE,
    'pca': PCA,
    'rf': SelectFromModel
}

# Helper function to create selectors from SELECTOR_MAP
def create_selectors_from_map(X_train, y_train, selector_map, **kwargs):
    selectors = {}
    for name in selector_map:
        selectors[name] = create_selector(name, X_train, y_train, **kwargs)
    return selectors

def create_selectors(X_train, y_train):
    selectors = {
        'rfe': create_selector('rfe'),
        'pca': create_selector('pca'),
        'rf': create_selector('rf', X_train, y_train)
    }
    return selectors

# Factory function to create selectors
def create_selector(method, X_train=None, y_train=None, **kwargs):
    if method not in SELECTOR_MAP:
        raise ValueError(f"Unknown method: {method}")
    
    if method == 'rfe':
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        return RFE(estimator, n_features_to_select=kwargs.get('n_features_to_select', 10))
    elif method == 'pca':
        return PCA(n_components=kwargs.get('n_components', 5))
    elif method == 'rf':
        estimator = RandomForestClassifier(n_estimators=100, random_state=0)
        estimator.fit(X_train, y_train)
        return SelectFromModel(estimator)

# Função para obter os espaços de busca para diferentes otimizações
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