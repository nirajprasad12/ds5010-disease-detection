# Library import
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Library import
import warnings
import os
import pandas as pd

import pkg_resources

# Could be any dot-separated package/module name or a "Requirement"
resource_package = __name__
resource_path = '/'.join(('datasets', 'diabetes.csv'))  
template = pkg_resources.resource_stream(resource_package, resource_path)


dataset = pd.read_csv(template)

X = dataset.drop(['Outcome'], axis=1)
Y = dataset['Outcome']

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

classifier_knn = KNeighborsClassifier(n_neighbors = 8, p = 1)
classifier_knn.fit(X_train, Y_train)

classifier_svc = SVC(kernel = 'poly', random_state = 0)
classifier_svc.fit(X_train, Y_train)

classifier_rf = RandomForestClassifier(n_estimators = 15, criterion = 'entropy', random_state = 0)
classifier_rf.fit(X_train, Y_train)

class diabetes:
    def __init__(self, X_inp):
        """Initialize the class"""
        self.flag = 0
        if X_inp is not None:
            for i in X_inp:
                if len(i) != 8:
                    self.flag = 1
                    print("Input array length must be 8 for all rows")
        
                if not all(isinstance(element, (int, float)) for element in i):
                    self.flag = 1
                    print("All elements in the input array must be numeric")
        else:
            self.flag = 1
            print("Input cannot be None")
        
        if self.flag == 0:
            self.X_inp = X_inp
    
    def KNearestNeighbours(self):
        if self.flag != 0:
            return 'Initialization failed'
        return list(classifier_knn.predict(self.X_inp)) 
    
    def SupportVectorClassifier(self):
        if self.flag != 0:
            return 'Initialization failed'
        return list(classifier_svc.predict(self.X_inp)) 
    
    def RandomForest(self):
        if self.flag != 0:
            return 'Initialization failed'
        return list(classifier_rf.predict(self.X_inp))


# test = [[6, 148, 72, 35, 0, 33.6, 0.627, 50],
# [11, 138, 76, 0, 0, 33.2, 0.42,	35],
# [10, 139, 80, 0, 0,	27.1, 1.441, 57]]

test = np.array(X_test)

g1 = diabetes(test)

accuracy_knn = accuracy_score(Y_test, g1.KNearestNeighbours())
accuracy_svc = accuracy_score(Y_test, g1.SupportVectorClassifier())
accuracy_rf = accuracy_score(Y_test, g1.RandomForest())

print(accuracy_knn, accuracy_svc, accuracy_rf)