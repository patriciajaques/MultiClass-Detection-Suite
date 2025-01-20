import argparse
from mnist.mnist_detection_pipeline import MNISTDetectionPipeline


def main():
    """Main function to run the MNIST detection pipeline."""
    print("Configurando argumentos do programa...")
    parser = argparse.ArgumentParser(
        description='Execute MNIST detection pipeline for specific stages')
    parser.add_argument('--start-stage', type=int,
                        help='Starting stage number')
    parser.add_argument('--end-stage', type=int,
                        help='Ending stage number')
    args = parser.parse_args()

    # Se foram fornecidos argumentos de intervalo, criar uma tupla com eles
    stage_range = None
    if args.start_stage is not None and args.end_stage is not None:
        stage_range = (args.start_stage, args.end_stage)
        print(f"Executando stages {stage_range[0]} até {stage_range[1]}")
    else:
        print("Executando todos os stages")

    print("Iniciando pipeline...")
    pipeline = MNISTDetectionPipeline(
        n_iter=50,
        n_jobs=6,
        test_size=0.2,
        stage_range=stage_range
    )

    print("Executando pipeline...")
    pipeline.run()
    print("Pipeline concluído!")

if __name__ == "__main__":
    print("Chamando função main()...")
    main()



#  /opt/anaconda3/envs/behavior_detection/bin/python /Users/patricia/Documents/code/python-code/behavior-detection/src/main_mnist.py --start-stage 6 --end-stage 6
