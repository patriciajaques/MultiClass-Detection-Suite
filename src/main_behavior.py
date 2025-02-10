import os
import warnings
from behavior.behavior_detection_pipeline import BehaviorDetectionPipeline

from core.utils import file_utils
from core.utils.path_manager import PathManager

def main():
    """Main function to run the behavior detection pipeline."""

    warnings.filterwarnings(
        "ignore",
        message=".*does not have valid feature names.*")

    PathManager.set_module('behavior')
    
    pipeline = BehaviorDetectionPipeline(
        n_iter=50, 
        n_jobs=6, 
        val_size=0.30,
        test_size=0.15,
        # pode ser optuna (default), grid ou random
        training_strategy_name='grid',
        use_voting_classifier=False
    )
    pipeline.run()

if __name__ == "__main__":
    os.system('clear')
    output_dir = "/Users/patricia/Documents/code/python-code/behavior-detection/output"
    file_utils.clear_output_directory(output_dir)
    main()
