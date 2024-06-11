import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple, Any

from feature_selection_params import get_param_grid

def create_rfe_selector(n_features_to_select: int) -> RFE:
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    return RFE(estimator, n_features_to_select=n_features_to_select)

def create_pca_selector(n_components: int) -> PCA:
    return PCA(n_components=n_components)

def create_rf_selector(X_train: np.ndarray, y_train: np.ndarray) -> SelectFromModel:
    estimator = RandomForestClassifier(n_estimators=100, random_state=0)
    estimator.fit(X_train, y_train)
    return SelectFromModel(estimator)

def create_pipeline(selector: BaseEstimator, classifier_params: Dict[str, Any] = {}) -> Pipeline:
    return Pipeline([
        ('feature_selection', selector),
        ('classifier', RandomForestClassifier(**classifier_params))
    ])

def evaluate_selectors(X_train: np.ndarray, y_train: np.ndarray, selectors: Dict[str, BaseEstimator], classifier_params: Dict[str, Any] = {'n_estimators': 100, 'random_state': 42}) -> Tuple[BaseEstimator, str, float]:
    best_score = -np.inf
    best_selector = None
    best_selector_name = ''

    for name, selector in selectors.items():
        pipeline = create_pipeline(selector, classifier_params)
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='balanced_accuracy')
        mean_score = scores.mean()

        if mean_score > best_score:
            best_score = mean_score
            best_selector = selector
            best_selector_name = name

    return best_selector, best_selector_name, best_score

def evaluate_feature_selector_with_search(X_train: np.ndarray, y_train: np.ndarray, selector: BaseEstimator, classifier_params: Dict[str, Any] = {'n_estimators': 100, 'random_state': 42}) -> Tuple[Pipeline, Dict[str, Any], float]:
    pipeline = create_pipeline(selector, classifier_params)
    
    # Obtenha os parâmetros válidos para o seletor específico
    if isinstance(selector, RFE):
        param_grid = {
            'feature_selection__n_features_to_select': [10, 20, 30, 40, 50]  # Ajuste conforme necessário
        }
    elif isinstance(selector, PCA):
        param_grid = {
            'feature_selection__n_components': [5, 10, 15, 20]  # Ajuste conforme necessário
        }
    else:
        param_grid = {}  # Outros seletores, se necessário

    search = GridSearchCV(pipeline, param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1)
    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_score = search.best_score_
    best_pipeline = search.best_estimator_

    best_selector = best_pipeline.named_steps['feature_selection']
    
    return best_selector, best_params, best_score

def evaluate_multiple_selectors_with_search(X_train: np.ndarray, y_train: np.ndarray, selectors: Dict[str, BaseEstimator], classifier_params: Dict[str, Any] = {'n_estimators': 100, 'random_state': 42}) -> Tuple[Pipeline, Dict[str, Any], float]:
    best_score = -np.inf
    best_selector = None
    best_selector_name = ''
    best_params = None
    
    # Avaliar e otimizar múltiplos seletores
    for name, selector in selectors.items():
        print(f"Evaluating selector: {name}")
        selector, params, score = evaluate_feature_selector_with_search(X_train, y_train, selector, classifier_params)
        
        if score > best_score:
            best_score = score
            best_selector = selector
            best_selector_name = name
            best_params = params

    print(f"Best selector: {best_selector_name} with params: {best_params}")
    return best_selector, best_params, best_score


def get_feature_selectors(X_train, y_train):
    # Definir seletores
    return {
        'RFE': create_rfe_selector(n_features_to_select=10),
        'PCA': create_pca_selector(n_components=5),
        'RF': create_rf_selector(X_train, y_train)
    }

class CustomSelector(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = 'rf', n_features: int = None, n_components: int = None):
        self.method = method
        self.n_features = n_features
        self.n_components = n_components
        self.selector = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CustomSelector':
        if self.method == 'rfe':
            self.selector = create_rfe_selector(self.n_features)
        elif self.method == 'pca':
            self.selector = create_pca_selector(self.n_components)
        elif self.method == 'rf':
            self.selector = create_rf_selector(X, y)
        self.selector.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.selector.transform(X)

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {"method": self.method, "n_features": self.n_features, "n_components": self.n_components}

    def set_params(self, **params) -> 'CustomSelector':
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
