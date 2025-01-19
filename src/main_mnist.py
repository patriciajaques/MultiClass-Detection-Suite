# src/main_mnist.py

import os
import sys
from pathlib import Path
import traceback

# Configuração inicial de path
current_file = Path(__file__).resolve()
src_dir = current_file.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    print("Iniciando programa...")
    print(f"Diretório atual: {os.getcwd()}")
    print(f"Python path: {sys.path}")

    from mnist.mnist_detection_pipeline import MNISTDetectionPipeline
    import argparse

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
        try:
            print("Chamando função main()...")
            main()
        except Exception as e:
            print(f"Erro durante a execução: {str(e)}")
            print("Traceback completo:")
            traceback.print_exc()
            sys.exit(1)
except Exception as e:
    print(f"Erro durante a importação: {str(e)}")
    print("Traceback completo:")
    traceback.print_exc()
    sys.exit(1)
