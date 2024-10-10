import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from fixtures import sample, sample_shorter
import pytest

# Módulo a ser testado
import preprocessing

def test_get_data_by_type(sample):

    # Verifique se a coluna 'log_type' está presente no DataFrame de entrada
    assert 'log_type' in sample.columns, "A coluna 'log_type' não está presente no DataFrame de entrada"

    result_categorical = preprocessing.get_data_by_type(sample, 'categorical', num_classes=5)
    
    # Adicione uma impressão para depuração
    print("Colunas do DataFrame result_categorical:", result_categorical.columns)
    
    assert 'log_type' in result_categorical.columns, "A coluna 'log_type' não está presente no DataFrame resultante"
    assert 'num_passos_total' not in result_categorical.columns
    
    result_numerical = preprocessing.get_data_by_type(sample, 'numerical', num_classes=5)
    
    # Adicione uma impressão para depuração
    print("Colunas do DataFrame result_numerical:", result_numerical.columns)
    
    assert 'num_passos_equacao' in result_numerical.columns, "A coluna 'num_passos_equacao' não está presente no DataFrame resultante"
    assert 'log_type' not in result_numerical.columns

def test_encode_single_column():
    data = pd.Series(['a', 'b', 'a', 'c'])
    encoded_data, le = preprocessing.encode_single_column(data)
    assert list(le.classes_) == ['a', 'b', 'c']
    assert all(encoded_data == [0, 1, 0, 2])

def test_encode_categorical_columns(sample):
    X_encoded, label_encoders = preprocessing.encode_categorical_columns(sample)
    
    # Verifique se a coluna 'log_type' está presente no DataFrame codificado
    assert 'log_type' in X_encoded.columns
    
    # Verifique se a coluna 'log_type' contém valores codificados entre 0 e 3
    assert set(X_encoded['log_type']).issubset({0, 1, 2, 3})
    
    assert 'log_type' in label_encoders

def test_apply_encoders_to_test_data(sample):
    X_encoded, label_encoders = preprocessing.encode_categorical_columns(sample)
    
    # Verifique se a coluna 'log_type' está presente no DataFrame codificado
    assert 'log_type' in X_encoded.columns
    
    # Verifique se a coluna 'log_type' tem um LabelEncoder
    assert 'log_type' in label_encoders

    # Verifique se a coluna 'log_type' contém valores codificados entre 0 e 3
    assert set(X_encoded['log_type']).issubset({0, 1, 2, 3})
    
    # Verifique se há pelo menos uma coluna categórica
    assert len(label_encoders) > 0

def test_load_data(mocker):
    mock_read_csv = mocker.patch('pandas.read_csv', return_value=pd.DataFrame({
        'comportamento': ['a', 'b', 'c'],
        'feature1': [1, 2, 3]
    }))
    mock_split_features_and_target = mocker.patch('utils.split_features_and_target', return_value=(pd.DataFrame({'feature1': [1, 2, 3]}), pd.DataFrame({'comportamento': ['a', 'b', 'c']})))
    X, y = preprocessing.load_data()
    assert X.shape == (3, 1)
    assert y.shape == (3,)

def test_split_train_test_data(sample):
    X_train, X_test, y_train, y_test = preprocessing.split_train_test_data(sample, sample['comportamento'])
    
    assert X_train.shape == (7, len(sample.columns) - 1)
    assert X_test.shape == (3, len(sample.columns) - 1)
    assert len(y_train) == 7
    assert len(y_test) == 3

def test_create_preprocessor(sample):
    X_train = sample.copy()
    preprocessor = preprocessing.create_preprocessor(X_train)
    
    assert isinstance(preprocessor, ColumnTransformer)
    assert len(preprocessor.transformers) == 2

def test_apply_smote(sample_shorter):
    X_train = sample_shorter[['feature1', 'feature2', 'feature3']]
    y_train = sample_shorter['target']
    
    # Aplicar SMOTE
    X_resampled, y_resampled = preprocessing.apply_smote(X_train, y_train)
    
    # Verificar se os dados foram balanceados
    assert len(X_resampled) > len(X_train)
    assert len(y_resampled) > len(y_train)
    
    # Verificar se a distribuição das classes está balanceada
    unique, counts = np.unique(y_resampled, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    assert class_distribution[0] == class_distribution[1]


