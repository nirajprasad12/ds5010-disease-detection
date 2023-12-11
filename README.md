## A Disease Detection Toolkit in Python

### Purpose:
At the core of our project is the development of an innovative Python package for disease detection. Drawing on advanced machine learning algorithms, we have built a comprehensive toolkit for the early identification of various illnesses, including cancer and diabetes. We have created a versatile and user-friendly solution that can be seamlessly integrated into existing systems and revolutionize healthcare in the process. Leveraging the power of Python libraries such as Pandas, NumPy, and Scikit-learn our package is an intuitive interface designed to facilitate efficient and accurate disease detection leading to improved patient outcomes.

Users who do not possess train and test data for either disease may use the main package functionality (```cancer.py, diabetes.py```) under ```/disease_detection```. For users who wish to train models using their own disease data, we have built a sub-package ```/manual_disease_detection``` that offers a variety of different features such as feature selection, data preprocessing and an option to train a Random Forest model using their data.

The entire package is published on PyPi and can be easily installed from here: https://pypi.org/project/disease-detection/

### Organization of Package Code:
```
/ds5010-disease-detection
  ├── /disease_detection
      ├── __init__.py
      ├── cancer.py
      ├── diabetes.py
      ├── /datasets
          └── cancer.csv
          └── diabetes.csv
      ├── /manual_disease_detection
          ├── __init__.py
          ├── extractfeatures.py
          ├── ml_model.py
          ├── preprocess.py
      ├── /tests
          ├── __init__.py
          ├── test_cancer.py
          ├── test_diabetes.py
  ├── README.md
  ├── setup.py
```
### Package Installation:
#### Software Requirements: 
- Python 3 and above
- PIP 22 and above
#### Steps:
- Open terminal and run ```pip install disease-detection```
- To upgrade, run ```pip install --upgrade disease-detection```

### Example: Cancer Detection
```ruby
# Import package/module on your Python env
from disease_detection import cancer

# 2 input records of 30 features each
inp_arr = [[13.54, 14.36, 87.46, 566.3,	0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462,	 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259],
[19.81,	22.15, 130, 1260, 0.09831, 0.1027, 0.1479, 0.09498, 0.1582, 0.05395, 0.7582, 1.017, 5.865, 112.4, 0.006494, 0.01893, 0.03391, 0.01521, 0.01356, 0.001997, 27.32, 30.88, 186.8, 2398, 0.1512, 0.315, 0.5372, 0.2388, 0.2768, 0.07615]]

# Instantiating an object of the class
g = cancer.cancer(inp_arr)

# Fitting models and printing out the classification of each input
# 0 indicates 'No Cancer' and 1 indicates 'Cancer'
print(g.LogisticRegression())
print(g.KNearestNeighbours())
print(g.SupportVectorClassifier())
print(g.GNB())
print(g.RandomForest())
```
Note that the length of the input array must be 30 for cancer, and all values should be numeric. Also, note that the input_array is a list of arrays - you can classify for any number of rows and our package should be able to handle this.

### Example: Diabetes Detection
```ruby
# Import package/module on your Python env
from disease_detection import diabetes

# 3 input records of 8 features each
inp_arr = [[6, 148, 72, 35, 0, 33.6, 0.627, 50],
[11, 138, 76, 0, 0, 33.2, 0.42,	35],
[10, 139, 80, 0, 0, 27.1, 1.441, 57]]

# Instantiating an object of the class
g = diabetes.diabetes(inp_arr)

# Fitting models and printing out the classification of each input
# 0 indicates 'No Diabetes' and 1 indicates 'Diabetes'
print(g.KNearestNeighbours())
print(g.SupportVectorClassifier())
print(g.RandomForest())
```
Note that the length of the input array must be 8 for diabetes, and all values should be numeric.

### Example: Manual Disease Detection

### Example: Pytests
We have set up a couple of simple unit tests using pytest to validate the input length, input data type and return data type of each classification model. You can run this in command line by simply navigating to the root of the repository ```ds5010-disease-detection/``` and running the following: 
```pytest -v disease_detection/tests```

