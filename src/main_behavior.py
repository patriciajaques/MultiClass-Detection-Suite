from behavior.behavior_detection_pipeline import BehaviorDetectionPipeline
import argparse

def main():
    """Main function to run the behavior detection pipeline."""
    # Adicionar argumentos para especificar o intervalo de stages
    # Exemplo: python src/main.py --start-stage 1 --end-stage 25
    parser = argparse.ArgumentParser(description='Execute behavior detection pipeline for specific stages')
    parser.add_argument('--start-stage', type=int, help='Starting stage number')
    parser.add_argument('--end-stage', type=int, help='Ending stage number')
    args = parser.parse_args()

    # Se foram fornecidos argumentos de intervalo, criar uma tupla com eles
    stage_range = None
    if args.start_stage is not None and args.end_stage is not None:
        stage_range = (args.start_stage, args.end_stage)

    pipeline = BehaviorDetectionPipeline(
        n_iter=50, 
        n_jobs=6, 
        test_size=0.2,
        stage_range=stage_range  # Passar o intervalo para o pipeline
    )
    pipeline.run()

if __name__ == "__main__":
    main()