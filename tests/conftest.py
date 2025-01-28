# tests/conftest.py
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_behavior_data():
    """Fixture com dados sintéticos para testes."""
    return pd.DataFrame({
        'aluno': [1, 1, 2, 2],
        'num_dia': [1, 1, 1, 1],
        'num_log': [1, 2, 1, 2],
        'comportamento': ['ON TASK', 'OFF TASK', 'ON TASK', 'ON SYSTEM'],
        'feature1': [0.1, 0.2, 0.3, 0.4],
        'feature2': [1.0, 2.0, 3.0, 4.0]
    })


@pytest.fixture
def sample_feature_matrix():
    """Fixture com matriz de features para testes."""
    return np.random.rand(100, 10)


@pytest.fixture
def sample_labels():
    """Fixture com labels para testes."""
    return np.random.choice(['ON TASK', 'OFF TASK', 'ON SYSTEM'], 100)
