from typing import List, Tuple, Dict, Any

from core.evaluation.evaluation import Evaluation
from core.feature_selection.feature_selection_factory import FeatureSelectionFactory
from core.pipeline.base_pipeline import BasePipeline
from core.training.optuna_bayesian_optimization_training import OptunaBayesianOptimizationTraining
from core.reporting import metrics_reporter


class StageTrainingManager:
    """Manages the execution of multiple training stages with different model and selector combinations."""

    def __init__(self, X_train, X_test, y_train, y_test, model_params,
                 n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_params = model_params
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

        # Initialize training strategy
        self.training_strategy = OptunaBayesianOptimizationTraining()

    def execute_stage(self, model_name: str, selector_name: str) -> Dict[str, Any]:
        """Executes a single training stage with specified model and selector.
        
        Args:
            model_name: Name of the model to train
            selector_name: Name of the feature selector to use
            
        Returns:
            Dict containing trained model and results
        """
        try:
            print(f"\nExecuting stage: {model_name} with {selector_name}")

            # Create pipeline
            pipeline = self._create_pipeline(model_name, selector_name)

            # Get selector search space if applicable
            selector_search_space = self._get_selector_search_space(selector_name)

            # Train the model
            results = self.training_strategy.train_model(
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
                selector_search_space=selector_search_space
            )

            return results
        except Exception as e:
            print(f"Error in stage {model_name} with {selector_name}: {str(e)}")
            return None


    def execute_all_stages(self, stages: List[Tuple[str, str, str]]):
        completed_stages = []
        failed_stages = []

        for stage_name, model_name, selector_name in stages:
            try:
                results = self.execute_stage(model_name, selector_name)
                if results:
                    completed_stages.append(stage_name)
                    self._generate_stage_reports(results, stage_name)
                else:
                    failed_stages.append(stage_name)
            except Exception as e:
                failed_stages.append(stage_name)
                print(f"Error in stage {stage_name}: {str(e)}")
                continue

        if completed_stages:
            print(f"\nCompleted stages: {', '.join(completed_stages)}")
        if failed_stages:
            print(f"\nFailed stages: {', '.join(failed_stages)}")

    def _create_pipeline(self, model_name: str, selector_name: str):
        """Creates a pipeline with specified model and selector."""
        # Get base model configuration
        model_config = self.model_params.get_models()[model_name]

        # Create feature selector if needed
        selector = None
        if selector_name != 'none':
            selector = FeatureSelectionFactory.create_selector(
                selector_name, self.X_train, self.y_train
            )

        return BasePipeline.create_pipeline(selector, model_config)

    def _get_selector_search_space(self, selector_name: str) -> Dict:
        """Gets search space for feature selector if applicable."""
        if selector_name == 'none':
            return {}

        selector = FeatureSelectionFactory.create_selector(
            selector_name, self.X_train, self.y_train
        )
        return selector.get_search_space()

    def _generate_stage_reports(self, results: Dict, stage_name: str):
        try:
            class_metrics, avg_metrics = Evaluation.evaluate_all_models(
                {stage_name: results},
                self.X_train, self.y_train,
                self.X_test, self.y_test
            )

            # Free memory after generating reports
            metrics_reporter.generate_reports(
                class_metrics,
                avg_metrics,
                filename_prefix=f"{stage_name}_"
            )
            del class_metrics
            del avg_metrics
        except Exception as e:
            print(f"Error generating reports for stage {stage_name}: {str(e)}")

    def _generate_final_reports(self, all_results: Dict):
        """Generates consolidated final reports."""
        print("\nGenerating final consolidated reports...")

        # Evaluate all models
        class_metrics, avg_metrics = Evaluation.evaluate_all_models(
            all_results,
            self.X_train, self.y_train,
            self.X_test, self.y_test
        )

        # Generate final reports
        metrics_reporter.generate_reports(
            class_metrics,
            avg_metrics,
            filename_prefix="_Final_",
            force_overwrite=True
        )

