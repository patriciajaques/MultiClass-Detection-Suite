import argparse
from mnist.mnist_detection_pipeline import MNISTDetectionPipeline


def main():
    """Main function to run the MNIST detection pipeline."""

    print("Iniciando pipeline...")
    pipeline = MNISTDetectionPipeline(
        n_iter=50,
        n_jobs=6,
        test_size=0.2
    )

    print("Executando pipeline...")
    pipeline.run()
    print("Pipeline concluído!")


if __name__ == "__main__":
    print("Chamando função main()...")
    main()


#  /opt/anaconda3/envs/behavior_detection/bin/python /Users/patricia/Documents/code/python-code/behavior-detection/src/main_mnist.py --start-stage 6 --end-stage 6
