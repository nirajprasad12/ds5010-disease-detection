import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import pkg_resources

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

resource_package = __name__
resource_path = '/'.join(('datasets', 'diabetes.csv'))  
template = pkg_resources.resource_stream(resource_package, resource_path)

dataset = pd.read_csv(template)

X = pd.DataFrame(dataset.drop(['Outcome'], axis=1).values)
Y = dataset['Outcome']

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
                    raise ValueError("Input array length must be 8 for all rows")
        
                if not all(isinstance(element, (int, float)) for element in i):
                    self.flag = 1
                    raise ValueError("All elements in the input array must be numeric")
        else:
            self.flag = 1
            raise ValueError("Input cannot be None")
        
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