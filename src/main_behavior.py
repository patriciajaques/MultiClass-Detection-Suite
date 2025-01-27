from behavior.behavior_detection_pipeline import BehaviorDetectionPipeline
import argparse

def main():
    """Main function to run the behavior detection pipeline."""

    pipeline = BehaviorDetectionPipeline(
        n_iter=50, 
        n_jobs=6, 
        test_size=0.2
    )
    pipeline.run()

if __name__ == "__main__":
    main()