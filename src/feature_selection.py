import pandas as pd
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def create_rfe_selector(n_features_to_select):
    """
    Cria um seletor RFE com um estimador de Regressão Logística.

    Args:
        n_features_to_select (int): O número de recursos a serem selecionados.

    Returns:
        RFE: Um seletor RFE configurado.
    """
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features_to_select)
    return selector

def create_pca_selector(n_components):
    """
    Cria um seletor PCA.

    Args:
        n_components (int): O número de componentes principais a serem mantidos.

    Returns:
        PCA: Um seletor PCA configurado.
    """
    selector = PCA(n_components=n_components)
    return selector

def create_rf_selector(X_train, y_train):
    """
    Cria um seletor com base na importância dos recursos de um Random Forest.

    Returns:
        SelectFromModel: Um seletor configurado.
    """
    estimator = RandomForestClassifier(n_estimators=100, random_state=0)
    estimator.fit(X_train, y_train)
    selector = SelectFromModel(estimator)
    return selector

def evaluate_feature_selectors(X_train, y_train, n_features_to_select, n_components):
    selectors = {
        'RFE': create_rfe_selector(n_features_to_select=n_features_to_select),
        'PCA': create_pca_selector(n_components=n_components),
        'RandomForest': create_rf_selector(X_train, y_train)
    }
    print("testando os 3 seletores criados")
    
    best_score = -np.inf
    best_selector = None
    best_selector_name = ''

    for name, selector in selectors.items():
        pipeline = Pipeline([
            ('feature_selection', selector),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Fit the pipeline to ensure all components are initialized
        pipeline.fit(X_train, y_train)
        
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='balanced_accuracy')
        mean_score = scores.mean()

        if mean_score > best_score:
            best_score = mean_score
            best_selector = selector
            best_selector_name = name

    return best_selector, best_selector_name
