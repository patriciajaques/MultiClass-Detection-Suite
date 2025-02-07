from typing import List, Dict, Any
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from core.models.model_persistence import ModelPersistence


from core.reporting.classification_model_metrics import ClassificationModelMetrics
from core.evaluation.model_evaluator import ModelEvaluator

class VotingEnsembleBuilder:
    """
    Classe responsável por construir um ensemble usando VotingClassifier
    com os melhores modelos treinados.
    """

    def select_best_models(metrics_df: pd.DataFrame,
                        n_models: int = 3,
                        metric: str = 'balanced_accuracy-val') -> List[str]:
        """
        Seleciona os melhores modelos base (excluindo ensembles anteriores)
        baseado na métrica especificada.

        Args:
            metrics_df: DataFrame com as métricas de todos os modelos
            n_models: Número de modelos a selecionar
            metric: Métrica para ordenar os modelos

        Returns:
            Lista com os nomes dos melhores modelos base
        """
        # Filtra ensembles anteriores
        base_models_df = metrics_df[~metrics_df['Model'].str.startswith('Voting_')]
        
        # Ordena pelo valor da métrica em ordem decrescente
        sorted_models = base_models_df.sort_values(by=metric, ascending=False)
        
        # Extrai o tipo base do modelo (parte antes do '_')
        sorted_models['base_model'] = sorted_models['Model'].apply(
            lambda x: x.split('_')[0])
        
        # Seleciona o melhor modelo de cada tipo
        best_models = []
        used_base_models = set()
        
        for _, row in sorted_models.iterrows():
            base_model = row['base_model']
            if base_model not in used_base_models:
                best_models.append(row['Model'])
                used_base_models.add(base_model)
                
                if len(best_models) == n_models:
                    break
                    
        return best_models

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
            stage_name,
            use_voting_classifier=True
        )
        ModelPersistence.save_model(voting_classifier, stage_name)
        return metrics

    @staticmethod
    def create_voting_ensemble(metrics_df: pd.DataFrame,
                               n_models: int = 3,
                               metric: str = 'balanced_accuracy-test',
                               voting: str = 'soft',
                               manual_selection: List[str] = None) -> Dict[str, Any]:
        """
        Cria um ensemble de voting de duas formas:
          1) Se manual_selection for None, os melhores modelos serão selecionados automaticamente
             com base na métrica definida no DataFrame metrics_df.
          2) Se manual_selection for fornecido (lista de nomes de modelos), essa lista será utilizada.
        
        Retorna:
          Um dicionário contendo:
            - 'ensemble': o objeto VotingClassifier criado
            - 'selected_models': a lista de nomes dos modelos usados
            - 'selection_metric': a métrica utilizada para seleção (se automática)
            - 'voting_type': o tipo de votação configurado
        """
        # Se for seleção manual, utilize a lista fornecida; caso contrário, selecione automaticamente.
        if manual_selection is None:
            selected_models = VotingEnsembleBuilder.select_best_models(
                metrics_df, n_models, metric)
        else:
            selected_models = manual_selection

        # Constrói o VotingClassifier com os modelos selecionados.
        voting_clf = VotingEnsembleBuilder.build_voting_classifier(
            selected_models, voting)

        return {
            'ensemble': voting_clf,
            'selected_models': selected_models,
            'selection_metric': metric if manual_selection is None else 'manual',
            'voting_type': voting
        }

    @staticmethod
    def _evaluate_ensemble_cv(ensemble, X, y, cv=5, scoring='balanced_accuracy'):
        scores = cross_val_score(ensemble, X, y, cv=cv, scoring=scoring)
        return scores.mean()
