from behavior.behavior_detection_pipeline import BehaviorDetectionPipeline

from core.utils.path_manager import PathManager

def main():
    """Main function to run the behavior detection pipeline."""

    PathManager.set_module('behavior')
    
    pipeline = BehaviorDetectionPipeline(
        n_iter=50, 
        n_jobs=6, 
        test_size=0.2
    )
    pipeline.run()

if __name__ == "__main__":
    main()