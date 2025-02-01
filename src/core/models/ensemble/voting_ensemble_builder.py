from typing import List, Dict, Any
import pandas as pd
from sklearn.ensemble import VotingClassifier
from core.models.model_persistence import ModelPersistence


from core.reporting.classification_model_metrics import ClassificationModelMetrics
from core.evaluation.model_evaluator import ModelEvaluator

class VotingEnsembleBuilder:
    """
    Classe responsável por construir um ensemble usando VotingClassifier
    com os melhores modelos treinados.
    """

    @staticmethod
    def select_best_models(metrics_df: pd.DataFrame,
                           n_models: int = 3,
                           metric: str = 'balanced_accuracy-val') -> List[str]:
        """
        Seleciona os n melhores modelos baseado na métrica especificada.

        Args:
            metrics_df: DataFrame com as métricas de todos os modelos
            n_models: Número de modelos a selecionar
            metric: Métrica para ordenar os modelos

        Returns:
            Lista com os nomes dos melhores modelos
        """
        # Ordena pelo valor da métrica em ordem decrescente
        sorted_models = metrics_df.sort_values(by=metric, ascending=False)

        # Retorna os n primeiros nomes de modelos
        return sorted_models['Model'].head(n_models).tolist()

    @staticmethod
    def build_voting_classifier(best_model_names: List[str],
                                voting: str = 'soft') -> VotingClassifier:
        """
        Constrói um VotingClassifier com os melhores modelos.

        Args:
            best_model_names: Lista com nomes dos melhores modelos
            voting: Tipo de votação ('hard' ou 'soft')

        Returns:
            VotingClassifier configurado com os melhores modelos
        """
        estimators = []

        for model_name in best_model_names:
            # Carrega o modelo salvo
            model = ModelPersistence.load_model(model_name)
            # Adiciona o pipeline completo como estimador
            estimators.append((model_name, model))

        return VotingClassifier(estimators=estimators, voting=voting)

    @staticmethod
    def train_and_evaluate_ensemble(voting_classifier: VotingClassifier,
                                    X_train, X_val, X_test, y_train, y_val, y_test,
                                    stage_name: str = "voting_ensemble") -> ClassificationModelMetrics:
        """
        Treina e avalia o ensemble.

        Args:
            voting_classifier: VotingClassifier configurado
            X_train, X_test: Features de treino e teste
            y_train, y_test: Labels de treino e teste
            stage_name: Nome para identificação do ensemble

        Returns:
            ClassificationModelMetrics com os resultados da avaliação
        """
        # Treina o ensemble
        print("\nTreinando Voting Ensemble...")
        voting_classifier.fit(X_train, y_train)

        # Avalia usando o ModelEvaluator existente
        metrics = ModelEvaluator.evaluate_single_model(
            voting_classifier,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            stage_name
        )
        ModelPersistence.save_model(voting_classifier, stage_name)
        return metrics

    @staticmethod
    def create_voting_ensemble(metrics_df: pd.DataFrame,
                               n_models: int = 3,
                               metric: str = 'balanced_accuracy-test',
                               voting: str = 'soft') -> Dict[str, Any]:
        """
        Cria um ensemble completo selecionando os melhores modelos.

        Args:
            metrics_df: DataFrame com métricas dos modelos
            n_models: Número de modelos para o ensemble
            metric: Métrica para seleção dos modelos
            voting: Tipo de votação

        Returns:
            Dicionário com o ensemble e informações dos modelos selecionados
        """
        # Seleciona os melhores modelos
        best_models = VotingEnsembleBuilder.select_best_models(
            metrics_df, n_models, metric)

        # Cria o voting classifier
        voting_clf = VotingEnsembleBuilder.build_voting_classifier(
            best_models, voting)

        return {
            'ensemble': voting_clf,
            'selected_models': best_models,
            'selection_metric': metric,
            'voting_type': voting
        }
