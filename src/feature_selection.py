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

# Function to create pipeline
def create_pipeline(selector, classifier_params={}):
    return Pipeline([
        ('feature_selection', selector),
        ('classifier', RandomForestClassifier(**classifier_params))
    ])

# Function to get parameter grid
def get_param_grid(selector, n_features):
    param_grid_map = {
        RFE: {
            'feature_selection__n_features_to_select': [
                n for n in [10, 20, 30, 40, 50] if n <= n_features
            ]
        },
        PCA: {
            'feature_selection__n_components': [
                n for n in [5, 10, 15, 20] if n <= n_features
            ]
        }
    }
    return param_grid_map.get(type(selector), {})

# Function to evaluate a selector with grid search
def evaluate_a_feature_selector_with_search(X_train, y_train, selector, cv=10, classifier_params={'n_estimators': 100, 'random_state': 42}):
    n_features = X_train.shape[1]  # Número de features no dataset
    pipeline = create_pipeline(selector, classifier_params)
    param_grid = get_param_grid(selector, n_features)  # Passar n_features
    search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_

# Function to evaluate multiple selectors with grid search
def evaluate_multiple_feature_selectors_with_search(X_train, y_train, selectors, cv=10, classifier_params={'n_estimators': 100, 'random_state': 42}):
    best_score = -np.inf
    best_selector = None
    best_selector_name = ''
    best_params = {}

    for name, selector in selectors.items():
        best_estimator, best_selector_params, mean_score = evaluate_a_feature_selector_with_search(X_train, y_train, selector, cv=cv, classifier_params=classifier_params)
        
        if mean_score > best_score:
            best_score = mean_score
            best_selector = best_estimator.named_steps['feature_selection']  # Extracting only the selector
            best_selector_name = name
            best_params = best_selector_params

    return best_selector, best_selector_name, best_params, best_score

# Example usage
if __name__ == "__main__":
    # Dummy data for example
    # Configuração dos parâmetros para geração de dados
    n_samples = 500   # Número de instâncias
    n_features = 20   # Número de features
    n_classes = 5      # Número de classes para classificação

    # Geração de dados sintéticos
    X_train = np.random.rand(n_samples, n_features)
    y_train = np.random.randint(0, n_classes, n_samples)

    selectors = {
        'pca': create_selector('pca'),
        'rfe': create_selector('rfe'),
        'rf': create_selector('rf', X_train, y_train)
    }

    best_selector, best_name, best_params, best_score = evaluate_multiple_feature_selectors_with_search(X_train, y_train, selectors)
    print(f"Best selector: {best_name} with params: {best_params} and score: {best_score}")