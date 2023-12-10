import numpy as np
import pytest
from disease_detection import diabetes
@pytest.fixture

def sample_input():
    return np.array([
        [6, 148, 72, 35, 0, 33.6, 0.627, 50],
        [11, 138, 76, 0, 0, 33.2, 0.42, 35],
        [10, 139, 80, 0, 0, 27.1, 1.441, 57]
    ])

def test_diabetes_initialization():
    with pytest.raises(ValueError):
        # Test initialization with None
        g = diabetes.diabetes(None)

def test_diabetes_input_validation(sample_input):
    with pytest.raises(ValueError):
        # Test initialization with invalid input length
        g = diabetes.diabetes(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]]))
    
    with pytest.raises(ValueError):
        # Test initialization with non-numeric input
        g = diabetes.diabetes(np.array([['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']]))

def test_diabetes_knn_prediction(sample_input):
    g = diabetes.diabetes(sample_input)
    result = g.KNearestNeighbours()
    assert isinstance(result, list)

def test_diabetes_svc_prediction(sample_input):
    g = diabetes.diabetes(sample_input)
    result = g.SupportVectorClassifier()
    assert isinstance(result, list)

def test_diabetes_rf_prediction(sample_input):
    g = diabetes.diabetes(sample_input)
    result = g.RandomForest()
    assert isinstance(result, list)