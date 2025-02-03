import traceback
from typing import List, Tuple, Dict, Any

import pandas as pd

from core.evaluation.model_evaluator import ModelEvaluator
from core.feature_selection.feature_selection_factory import FeatureSelectionFactory
from core.management.progress_tracker import ProgressTracker
from core.models.ensemble.voting_ensemble_builder import VotingEnsembleBuilder
from core.models.model_persistence import ModelPersistence
from core.pipeline.base_pipeline import BasePipeline
from core.reporting.classification_model_metrics import ClassificationModelMetrics
from core.reporting.metrics_persistence import MetricsPersistence
from core.training.optuna_bayesian_optimization_training import OptunaBayesianOptimizationTraining
from core.reporting.metrics_reporter import MetricsReporter


class StageTrainingManager:
    """Manages the execution of multiple training stages with different model and selector combinations."""

    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test,  model_params,
                 n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1, training_strategy=None, use_voting_classifier=True):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.model_params = model_params
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.use_voting_classifier = use_voting_classifier

        # Initialize training strategy
        self.training_strategy = training_strategy or OptunaBayesianOptimizationTraining()
        self.progress_tracker = ProgressTracker()

    def execute_stage(self, model_name: str, selector_name: str) -> ClassificationModelMetrics:
        """Executes a single training stage with specified model and selector.

        Args:
            model_name: Name of the model to train
            selector_name: Name of the feature selector to use

        Returns:
            Dict containing trained model and result
        """

        stage_name = f"{model_name}_{selector_name}"

        try:
            # Create pipeline
            pipeline = self._create_pipeline(model_name, selector_name)

            # Get selector search space if applicable
            selector_search_space = self._get_selector_search_space(
                selector_name)

            # Train the model
            trained_model_info = self.training_strategy.train_model(
                pipeline=pipeline,
                X_train=self.X_train,
                y_train=self.y_train,
                model_name=model_name,
                model_params=self.model_params,
                selector_name=selector_name,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                selector_search_space=selector_search_space,
            )

            # Evaluate the model
            model_metrics = ModelEvaluator.evaluate_single_model(
                trained_model_info['pipeline'], 
                self.X_train, self.y_train, self.X_val, 
                self.y_val, self.X_test, self.y_test, 
                stage_name, self.use_voting_classifier
            )
            model_metrics.execution_time = trained_model_info['execution_time']
            model_metrics.training_type = trained_model_info['training_type']
            model_metrics.hyperparameters = trained_model_info['hyperparameters']
            model_metrics.cv_score = trained_model_info['cv_score']

            # Generate reports
            MetricsReporter.generate_stage_report(model_metrics)

            # Save trained model
            ModelPersistence.save_model(
                trained_model_info['pipeline'], stage_name)

            return model_metrics
        except Exception as e:
            print(f"Error in {self.__class__.__name__}.execute_stage: {model_name} with {selector_name}. "
                  f"Exception: {str(e)}. Traceback: {traceback.format_exc()}")
            return None

    def execute_all_stages(self, stages: List[Tuple[str, str, str]]):
        completed_stages = []
        failed_stages = []
        all_metrics = []

        print("\n=== Executando todos os estágios ===")
        print("\nEstágios a serem executados:")
        stage_names = [
            f"{model_name} with {selector_name}" for model_name, selector_name in stages]
        print("\n" + ", ".join(stage_names) + "\n")

        for model_name, selector_name in stages:
            stage_name = f"{model_name}_{selector_name}"
            try:
                # Check if stage has already been completed
                if self.progress_tracker.is_completed(stage_name):
                    print(f"Stage {stage_name} already completed. Skipping...")
                    model_metrics = MetricsPersistence.load_metrics(stage_name)
                    all_metrics.append(model_metrics)
                else:
                    print(f"\n=== Executando estágio: {model_name} com {selector_name} ===")
                    model_metrics = self.execute_stage(model_name, selector_name)
                    if model_metrics is not None:
                        MetricsPersistence.save_metrics(model_metrics, stage_name)
                        completed_stages.append(stage_name)
                        self.progress_tracker.save_progress(stage_name)
                        all_metrics.append(model_metrics)
                    else:
                        failed_stages.append(stage_name)
                        continue
            except Exception as e:
                failed_stages.append(stage_name)
                print(f"Error in stage {stage_name}: {str(e)}")
                continue

        if self.use_voting_classifier:
            # Cria o ensemble
            ensemble_metrics = self._create_and_evaluate_ensemble(all_metrics)
        
            # Adiciona as métricas do ensemble ao relatório final
            all_metrics.append(ensemble_metrics)

        # Gera o relatório final com os modelos base e ensemble 
        MetricsReporter.generate_final_report(all_metrics)
        self._print_execution_summary(completed_stages, failed_stages)

    def _create_and_evaluate_ensemble(self, all_metrics):
        ensemble_info = VotingEnsembleBuilder.create_voting_ensemble(
            MetricsReporter.assemble_metrics_summary(all_metrics),
            n_models=3,
            metric='balanced_accuracy-test',
            voting='soft'
        )

        print("\nEnsemble criado com os seguintes modelos:")
        for model in ensemble_info['selected_models']:
            print(f"- {model}")

        # Treina e avalia o ensemble
        ensemble_metrics = VotingEnsembleBuilder.train_and_evaluate_ensemble(
            ensemble_info['ensemble'],
            self.X_train, self.X_val, self.X_test,
            self.y_train, self.y_val, self.y_test,
            stage_name="voting_ensemble"
        )
        
        return ensemble_metrics    

    def _create_pipeline(self, model_name: str, selector_name: str):
        """Creates a pipeline with specified model and selector."""
        # Get base model configuration
        model_config = self.model_params.get_models()[model_name]

        # Create feature selector if needed
        selector = None
        if selector_name != 'none':
            selector = selector = FeatureSelectionFactory.create_selector(
                method=selector_name
            )

        return BasePipeline.create_pipeline(selector, model_config)

    def _get_selector_search_space(self, selector_name: str) -> Dict:
        """Gets search space for feature selector if applicable."""
        if selector_name == 'none':
            return {}

        selector = selector = FeatureSelectionFactory.create_selector(
            method=selector_name
        )
        return selector.get_search_space()


    def _print_execution_summary(self, completed_stages, failed_stages):
        print("\n=== Sumário da Execução ===")
        if completed_stages:
            print(
                f"\nEstágios completados nesta execução: {', '.join(completed_stages)}")
        if failed_stages:
            print(f"\nEstágios com falha: {', '.join(failed_stages)}")

        all_completed = self.progress_tracker.completed_pairs
        print(f"\nTotal de estágios já completados: {len(all_completed)}")
        print(f"Estágios completados: {', '.join(all_completed)}")
