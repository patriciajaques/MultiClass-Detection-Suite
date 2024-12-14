from behavior.behavior_detection_pipeline import BehaviorDetectionPipeline


def main():
    """Main function to run the behavior detection pipeline."""
    pipeline = BehaviorDetectionPipeline(n_iter=50, n_jobs=-1, test_size=0.2)
    pipeline.run()

if __name__ == "__main__":
    main()