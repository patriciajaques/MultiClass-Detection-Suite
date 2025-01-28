import pytest
from core.preprocessors.data_cleaner import DataCleaner
from core.config.config_manager import ConfigManager


def test_remove_instances_with_value(sample_behavior_data):
    cleaner = DataCleaner()
    cleaned_data = cleaner.remove_instances_with_value(
        sample_behavior_data, 'comportamento', 'OFF TASK')

    assert 'OFF TASK' not in cleaned_data['comportamento'].values
    assert len(cleaned_data) == len(sample_behavior_data) - 1


def test_remove_columns_with_config(sample_behavior_data):
    config_manager = ConfigManager()
    cleaner = DataCleaner(config_manager)

    cleaned_data = cleaner.remove_columns(
        sample_behavior_data, use_config=True)

    # Verifica se as colunas configuradas foram removidas
    assert 'id_log' not in cleaned_data.columns
    assert 'grupo' not in cleaned_data.columns


@pytest.mark.parametrize("undefined_value", ['?', 'undefined', None])
def test_handle_undefined_values(sample_behavior_data, undefined_value):
    cleaner = DataCleaner()
    data_with_undefined = sample_behavior_data.copy()
    data_with_undefined.loc[0, 'comportamento'] = undefined_value

    cleaned_data = cleaner.remove_instances_with_value(
        data_with_undefined, 'comportamento', undefined_value)

    assert undefined_value not in cleaned_data['comportamento'].values
