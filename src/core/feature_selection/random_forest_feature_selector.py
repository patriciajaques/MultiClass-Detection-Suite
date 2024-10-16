import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from core.feature_selection.base_feature_selector import BaseFeatureSelector

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomForestFeatureSelector(BaseFeatureSelector):
    def __init__(self, X_train, y_train):
        super().__init__(X_train, y_train)

    def _create_selector(self):
        estimator = RandomForestClassifier(n_estimators=100, random_state=0)
        estimator.fit(self.X_train, self.y_train)
        selector = SelectFromModel(estimator)
        selector.fit(self.X_train, self.y_train)

        initial_features = selector.get_support().sum()
        logger.info(f"Inicialmente selecionadas {initial_features} características.")

        if initial_features == 0:
            for percentile in [75, 50, 25, 10, 5]:  # Adicionamos percentis mais baixos
                threshold = np.percentile(selector.estimator_.feature_importances_, percentile)
                selector.threshold_ = threshold
                selected_features = selector.get_support().sum()
                logger.info(f"Ajuste do limiar para o percentil {percentile}, novas características selecionadas: {selected_features}")
                if selected_features > 0:
                    break

            # Se ainda não tiver selecionado nenhuma característica, selecione pelo menos uma
            if selector.get_support().sum() == 0:
                max_importance = selector.estimator_.feature_importances_.max()
                selector.threshold_ = max_importance * 0.9
                logger.info("Forçando a seleção de pelo menos uma característica")

        final_features = selector.get_support().sum()
        logger.info(f"Finalmente selecionadas {final_features} características.")

        return selector


    def get_search_space(self):
        return {'feature_selection__threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}