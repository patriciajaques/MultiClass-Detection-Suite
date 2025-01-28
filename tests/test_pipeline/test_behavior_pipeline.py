import pytest
from behavior.behavior_detection_pipeline import BehaviorDetectionPipeline


def test_pipeline_initialization():
    pipeline = BehaviorDetectionPipeline(n_iter=10, n_jobs=1, test_size=0.2)
    assert pipeline.target_column == 'comportamento'
    assert pipeline.n_iter == 10
    assert pipeline.test_size == 0.2


def test_data_preparation(sample_behavior_data):
    pipeline = BehaviorDetectionPipeline(n_iter=10, n_jobs=1, test_size=0.2)
    X_train, X_test, y_train, y_test = pipeline.prepare_data(
        sample_behavior_data)

    # Verifica se os dados foram divididos corretamente
    assert len(X_train) + len(X_test) == len(sample_behavior_data)
    assert 'comportamento' not in X_train.columns
    assert all(y_train.isin(['ON TASK', 'OFF TASK', 'ON SYSTEM']))


@pytest.mark.slow
def test_complete_pipeline_execution(sample_behavior_data):
    pipeline = BehaviorDetectionPipeline(n_iter=1, n_jobs=1, test_size=0.2)
    try:
        pipeline.run()
        success = True
    except Exception as e:
        success = False
    assert success
