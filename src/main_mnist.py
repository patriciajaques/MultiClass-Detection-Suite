import os
import warnings
import core.utils.file_utils  as file_utils
from core.utils.path_manager import PathManager
from mnist.mnist_detection_pipeline import MNISTDetectionPipeline


def main():
    """Main function to run the MNIST detection pipeline."""

    warnings.filterwarnings(
        "ignore",
        message=".*does not have valid feature names.*")

    PathManager.set_module('mnist')

    pipeline = MNISTDetectionPipeline(
        target_column='target',
        n_iter=50,
        n_jobs=6,
        val_size=0.25,
        test_size=0.2,
        training_strategy_name='optuna', # pode ser optuna (default), grid ou random
        use_voting_classifier=True
    )

    print("Executando pipeline...")
    pipeline.run()


if __name__ == "__main__":
    os.system('clear')
    # print("Chamando função main()...")
    # output_dir = "/Users/patricia/Documents/code/python-code/behavior-detection/output"
    # file_utils.clear_output_directory(output_dir)
    main()