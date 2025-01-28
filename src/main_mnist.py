from core.utils.path_manager import PathManager
from mnist.mnist_detection_pipeline import MNISTDetectionPipeline


def main():
    """Main function to run the MNIST detection pipeline."""

    PathManager.set_module('mnist')

    pipeline = MNISTDetectionPipeline(
        target_column='target',
        n_iter=50,
        n_jobs=6,
        test_size=0.2
    )

    print("Executando pipeline...")
    pipeline.run()


if __name__ == "__main__":
    print("Chamando função main()...")
    main()

