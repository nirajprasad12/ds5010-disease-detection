import pytest
import numpy as np

from disease_detection import cancer  # Replace 'your_module' with the actual module name containing the 'cancer' class


@pytest.fixture
def sample_input():
    return [
        [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56,
         0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288,
         0.2977, 0.07259],
        [19.81, 22.15, 130, 1260, 0.09831, 0.1027, 0.1479, 0.09498, 0.1582, 0.05395, 0.7582, 1.017, 5.865, 112.4,
         0.006494, 0.01893, 0.03391, 0.01521, 0.01356, 0.001997, 27.32, 30.88, 186.8, 2398, 0.1512, 0.315, 0.5372, 0.2388,
         0.2768, 0.07615]
    ]

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cancer_initialization():
    with pytest.raises(ValueError):
        # Test initialization with None
        g = cancer.cancer(None)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cancer_input_validation(sample_input):
    with pytest.raises(ValueError):
        # Test initialization with invalid input length
        g = cancer.cancer(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]]))
    
    with pytest.raises(ValueError):
        # Test initialization with non-numeric input
        g = cancer.cancer(np.array([['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']]))

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cancer_knn_prediction(sample_input):
    g = cancer.cancer(sample_input)
    result = g.KNearestNeighbours()
    assert isinstance(result, list)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cancer_logistic_prediction(sample_input):
    g = cancer.cancer(sample_input)
    result = g.LogisticRegression()
    assert isinstance(result, list)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cancer_gnb_prediction(sample_input):
    g = cancer.cancer(sample_input)
    result = g.GNB()
    assert isinstance(result, list)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cancer_svc_prediction(sample_input):
    g = cancer.cancer(sample_input)
    result = g.SupportVectorClassifier()
    assert isinstance(result, list)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cancer_rf_prediction(sample_input):
    g = cancer.cancer(sample_input)
    result = g.RandomForest()
    assert isinstance(result, list)
