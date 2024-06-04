from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def create_rfe_selector(n_features_to_select):
    """
    Cria um seletor RFE com um estimador de Regressão Logística.

    Args:
        n_features_to_select (int): O número de recursos a serem selecionados.

    Returns:
        RFE: Um seletor RFE configurado.
    """
    estimator = LogisticRegression()
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

def create_rf_selector():
    """
    Cria um seletor com base na importância dos recursos de um Random Forest.

    Returns:
        SelectFromModel: Um seletor configurado.
    """
    estimator = RandomForestClassifier(n_estimators=100, random_state=0)
    selector = SelectFromModel(estimator)
    return selector