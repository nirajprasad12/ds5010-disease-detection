import numpy as np
import pytest
from disease_detection import heart 
@pytest.fixture

def sample_input():
    return np.array([
    
    [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1],  # Sample 1
    [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 1, 3, 2],  # Sample 2
    [62, 0, 4, 140, 268, 0, 2, 160, 0, 3.6, 3, 2, 3]   # Sample 3
        
    ])

def test_heart_initialization():
    with pytest.raises(ValueError):
        # Test initialization with None
        h = heart.HeartDisease(None)

def test_heart_input_validation(sample_input):
    with pytest.raises(ValueError):
        # Test initialization with invalid input length
        h = heart.HeartDisease(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]]))
    
    with pytest.raises(ValueError):
        # Test initialization with non-numeric input
        h = heart.HeartDisease(np.array([[63, 1, 3, 145, 233, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']]))

def test_heart_knn_prediction(sample_input):
    h = heart.HeartDisease(sample_input)
    result = h.KNearestNeighbours()
    assert isinstance(result, list)

def test_heart_svc_prediction(sample_input):
    h = heart.HeartDisease(sample_input)
    result = h.SupportVectorClassifier()
    assert isinstance(result, list)

def test_heart_rf_prediction(sample_input):
    h = heart.HeartDisease(sample_input)
    result = h.RandomForest()
    assert isinstance(result, list)

