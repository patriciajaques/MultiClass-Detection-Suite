import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from fixtures import sample, sample_shorter
import pytest
from io import StringIO

# Módulo a ser testado
import behavior.utils as utils
from data_processor import BaseDataProcessor

@pytest.fixture
def processor():
    return BaseDataProcessor()

@pytest.fixture
def sample_csv():
    csv_data = """col1;log_type;comportamento
            1;2;A
            3;4;B
            5;6;C"""
    return pd.read_csv(StringIO(csv_data), delimiter=';')

def test_load_data(monkeypatch, processor, sample_csv):
    # Mock do método pd.read_csv para retornar o DataFrame de teste
    def mock_read_csv(file_path, delimiter):
        return sample_csv
    
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    
    df = processor.load_data()
    pd.testing.assert_frame_equal(df, sample_csv)

def test_split_data(processor, sample):
    X, y = processor.split_data(sample, 'comportamento')
    expected_X = sample.drop(columns=['comportamento'])
    expected_y = sample['comportamento']
    pd.testing.assert_frame_equal(X, expected_X)
    pd.testing.assert_series_equal(y, expected_y)

def test_get_data_by_type(processor, sample):

    # Verifique se a coluna 'log_type' está presente no DataFrame de entrada
    assert 'log_type' in sample.columns, "A coluna 'log_type' não está presente no DataFrame de entrada"

    result_categorical = processor.get_data_by_type(sample, 'categorical', num_classes=5)
    
    # Adicione uma impressão para depuração
    # print("Colunas do DataFrame result_categorical:", result_categorical.columns)
    
    assert 'log_type' in result_categorical.columns, "A coluna 'log_type' não está presente no DataFrame resultante"
    assert 'num_passos_total' not in result_categorical.columns
    
    result_numerical = processor.get_data_by_type(sample, 'numerical', num_classes=5)
    
    # Adicione uma impressão para depuração
    # print("Colunas do DataFrame result_numerical:", result_numerical.columns)
    
    assert 'num_passos_equacao' in result_numerical.columns, "A coluna 'num_passos_equacao' não está presente no DataFrame resultante"
    assert 'log_type' not in result_numerical.columns

def _test_encode_single_column():
    data = pd.Series(['a', 'b', 'a', 'c'])
    encoded_data, le = processor.encode_single_column(data)
    assert list(le.classes_) == ['a', 'b', 'c']
    assert all(encoded_data == [0, 1, 0, 2])

def _test_encode_categorical_columns(sample):
    X_encoded, label_encoders = processor.encode_categorical_columns(sample)
    
    # Verifique se a coluna 'log_type' está presente no DataFrame codificado
    assert 'log_type' in X_encoded.columns
    
    # Verifique se a coluna 'log_type' contém valores codificados entre 0 e 3
    assert set(X_encoded['log_type']).issubset({0, 1, 2, 3})
    
    assert 'log_type' in label_encoders

def _test_apply_encoders_to_test_data(sample):
    X_encoded, label_encoders = processor.encode_categorical_columns(sample)
    
    # Verifique se a coluna 'log_type' está presente no DataFrame codificado
    assert 'log_type' in X_encoded.columns
    
    # Verifique se a coluna 'log_type' tem um LabelEncoder
    assert 'log_type' in label_encoders

    # Verifique se a coluna 'log_type' contém valores codificados entre 0 e 3
    assert set(X_encoded['log_type']).issubset({0, 1, 2, 3})
    
    # Verifique se há pelo menos uma coluna categórica
    assert len(label_encoders) > 0

def _test_create_preprocessor(sample):
    X_train = sample.copy()
    preprocessor = processor.create_preprocessor(X_train)
    
    assert isinstance(preprocessor, ColumnTransformer)
    assert len(preprocessor.transformers) == 2

def _test_apply_smote(sample_shorter):
    X_train = sample_shorter[['feature1', 'feature2', 'feature3']]
    y_train = sample_shorter['target']
    
    # Aplicar SMOTE
    X_resampled, y_resampled = processor.apply_smote(X_train, y_train)
    
    # Verificar se os dados foram balanceados
    assert len(X_resampled) > len(X_train)
    assert len(y_resampled) > len(y_train)
    
    # Verificar se a distribuição das classes está balanceada
    unique, counts = np.unique(y_resampled, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    assert class_distribution[0] == class_distribution[1]