## A Disease Detection Toolkit in Python

### Purpose:
At the core of our project is the development of an innovative Python package for disease detection. Drawing on advanced machine learning algorithms, we have built a comprehensive toolkit for the early identification of various illnesses, including cancer and diabetes. We have created a versatile and user-friendly solution that can be seamlessly integrated into existing systems and revolutionize healthcare in the process. Leveraging the power of Python libraries such as Pandas, NumPy, and Scikit-learn our package is an intuitive interface designed to facilitate efficient and accurate disease detection leading to improved patient outcomes.

Users who do not possess train data for either disease, and simply wish to check if a particular record of data indicates the presence of the disease, may use the main package functionality (cancer.py, diabetes.py) under ```/disease_detection``` to classify their one or more records as “having the disease” or “not having the disease”. For users who wish to train models using their own disease data, we have built a sub-package ```/manual_disease_detection``` that offers a variety of different features such as feature selection, data preprocessing, an option to train and evaluate ML models using their train data and use the developed models to classify their own test data.


The entire package is published on PyPi and can be easily installed from here: https://pypi.org/project/disease-detection/

### Organization of Package Code:
```
/ds5010-disease-detection
  ├── /disease_detection
      ├── __init__.py
      ├── cancer.py
      ├── diabetes.py
      ├── heart.py
      ├── /datasets
          └── cancer.csv
          └── diabetes.csv
          └── heart.csv
      ├── /manual_disease_detection
          ├── __init__.py
          ├── extractfeatures.py
          ├── ml_model.py
          ├── preprocess.py
      ├── /tests
          ├── __init__.py
          ├── test_cancer.py
          ├── test_diabetes.py
          ├── test_heart.py
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

### Example: Heart Disease Detection
```ruby
# Import package/module on your Python env
from disease_detection import heart

# 3 input records of 13 features each
inp_arr = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1],
    [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 1, 3, 2],
    [62, 0, 4, 140, 268, 0, 2, 160, 0, 3.6, 3, 2, 3]]

# Instantiating an object of the class
g = heart.HeartDisease(inp_arr)

# Fitting models and printing out the classification of each input
# 0 indicates 'No Heart Disease' and 1 indicates 'Heart Disease'
print(g.KNearestNeighbours())
print(g.SupportVectorClassifier())
print(g.RandomForest())
```
Note that the length of the input array must be 13 for heart disease, and all values should be numeric.

### Example: Manual Disease Detection

To import and prepare the dataset to pass to our package's manual detection module, do the following:
```ruby
# import sub-package modules
from disease_detection.manual_disease_detection import preprocess
from disease_detection.manual_disease_detection import extractfeatures
from disease_detection.manual_disease_detection import ml_model

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# split X and y in your datasets:
dataset = pd.read_csv('cancer.csv')
dataset = dataset.iloc[:, :-1]
X = dataset.drop(['id', 'diagnosis'], axis=1)
Y = dataset['diagnosis']
labelencoder_Y = LabelEncoder()  # replace M and B with 0s and 1s
Y = labelencoder_Y.fit_transform(Y)
```

The next step (optional) uses our package preprocess to impute nulls with mean values:
```ruby
X = preprocess.preprocess_data(X)
```

This step (optional) selects the k-best features from your dataset X. The value of k cannot go beyond the max number of features in your dataset. We have used Anova statistic to fetch the best features:
```ruby
X = pd.DataFrame(extractfeatures.select_best_features(X, Y, 'all'))  # for all features
# X = pd.DataFrame(extractfeatures.select_best_features(X, Y, 5))  # for 5 best features
```
#### ML_Model.py

This is the outline showcasing all functions in the ML_model.py file:

<img width="324" alt="Screenshot 2023-12-11 at 5 11 56 PM" src="https://github.com/nirajprasad12/ds5010-disease-detection/assets/26063090/0e741c23-1984-427e-badb-279128b1af4b">

1. Now to train the required ML models, run the following:
```ruby
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)  # optional step to split dataset and train

model = ml_model.train_random_forest(X_train, Y_train)  # optional parameters: (n_estimators=100, max_depth=None)
# model = ml_model.train_logistic_regression(X_train, Y_train)  # optional parameters: (random_state = 0)
# model = ml_model.train_k_nearest_neighbours(X_train, y_train)  # optional parameters:  (n_neighbors = 5, metric = 'minkowski', p = 2)
```
2. Pass the model to predict values based on your data:
```ruby
y_pred = ml_model.predict_data(model, X_test)
```
3. You can use the evaluate_model function to evaluate the model's performance and find accuracy. The input parameters for this method are evaluate_model(clf, X_test, y_test, req_result = 0) where clf is the model instance fitted above, X_test and y_test are the evaluation input and expected output, and req_result is a flag.
- if req_result = 0 or left empty, then return all three y_pred, accuracy, report
- if req_result = 1, then return y_pred
- if req_result = 2, then return accuracy
- if req_result = 3, then return report
```ruby
y_pred, accuracy, report = ml_model.evaluate_model(model, X_test, Y_test)
# y_pred = ml_model.evaluate_model(model, X_test, Y_test, 1)
# accuracy = ml_model.evaluate_model(model, X_test, Y_test, 2)
# report = ml_model.evaluate_model(model, X_test, Y_test, 3)
```
4. You can also output a seaborn plot of the confusion matrix by passing y_pred (predicted) and y_true (expected) to the plot_confusion_matrix function:
```ruby
ml_model.plot_confusion_matrix(Y_test, y_pred)
```

### Example: Pytests
We have set up a couple of simple unit tests using pytest to validate the input length, input data type and return data type of each classification model. 
Currently, the package does not include tests to validate the correctness of the prediction itself.

If you download and install the package manually, you can run this in command line by simply navigating to the root of the repository ```ds5010-disease-detection/``` and running the following in CLI: 
```pytest -v disease_detection/tests```

![Screenshot 2023-12-11 at 4 28 44 PM](https://github.com/nirajprasad12/ds5010-disease-detection/assets/26063090/be078576-3d24-4169-9f40-dfdc385f04e2)

If you install the package on PIP and want to access and run the unit tests on a platform (VS Code/Google Colab/Jupyter), follow the below steps:

<img width="1252" alt="Screenshot 2023-12-10 at 8 37 39 PM" src="https://github.com/nirajprasad12/ds5010-disease-detection/assets/26063090/73733623-9788-49c1-90b1-243f2bdf0249">




