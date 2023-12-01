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

# Library import
import warnings
import os
import pandas as pd

import pkg_resources

# Could be any dot-separated package/module name or a "Requirement"
resource_package = __name__
resource_path = '/'.join(('datasets', 'cancer.csv'))  # Do not use os.path.join()
#template = pkg_resources.resource_string(resource_package, resource_path)
# or for a file-like stream:
template = pkg_resources.resource_stream(resource_package, resource_path)


dataset = pd.read_csv(template)
dataset = dataset.iloc[:, :-1]

X = dataset.drop(['id', 'diagnosis'], axis=1)
Y = dataset['diagnosis']

# (M = malignant, B = benign) 

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

class cancer:
    def __init__(self):
        """Initialize the perceptron with given weights (default is None)"""
        #self.X_inp = X_inp
        pass

    def LogisticRegression(self, X_inp):
        classifier_lr = LogisticRegression(random_state = 0)
        classifier_lr.fit(X_train, Y_train)
        return list(classifier_lr.predict(X_inp))
    
    def KNearestNeighbours(self, X_inp):
        classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier_knn.fit(X_train, Y_train)
        return list(classifier_knn.predict(X_inp)) 
    
    def SupportVectorClassifier(self, X_inp):
        classifier_svc = SVC(kernel = 'linear', random_state = 0)
        classifier_svc.fit(X_train, Y_train)
        return list(classifier_svc.predict(X_inp)) 
    
    def GNB(self, X_inp):
        classifier_nb = GaussianNB()
        classifier_nb.fit(X_train, Y_train)
        return list(classifier_nb.predict(X_inp)) 
    
    def RandomForest(self, X_inp):
        classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier_rf.fit(X_train, Y_train)
        return list(classifier_rf.predict(X_inp))

    
# g = cancer()

# print(g.LogisticRegression(X_test))
# print(g.KNearestNeighbours(X_test))
# print(g.SupportVectorClassifier(X_test))
# print(g.GNB(X_test))
# print(g.RandomForest(X_test))


# something = [[13.54,	14.36,	87.46,	566.3,	0.09779,	0.08129,	0.06664,	0.04781,	0.1885,	0.05766,	0.2699,	0.7886,	2.058,	23.56,	0.008462,	0.0146,	0.02387,	0.01315,	0.0198,	0.0023,	15.11,	19.26,	99.7,	711.2,	0.144,	0.1773,	0.239,	0.1288,	0.2977,	0.07259	],
# [19.81,	22.15,	130,	1260,	0.09831,	0.1027,	0.1479,	0.09498,	0.1582,	0.05395,	0.7582,	1.017,	5.865,	112.4,	0.006494,	0.01893,	0.03391,	0.01521,	0.01356,	0.001997,	27.32,	30.88,	186.8,	2398,	0.1512,	0.315,	0.5372,	0.2388,	0.2768,	0.07615	]]

# print(g.LogisticRegression(something))
# print(g.KNearestNeighbours(something))
# print(g.SupportVectorClassifier(something))
# print(g.GNB(something))
# print(g.RandomForest(something))

