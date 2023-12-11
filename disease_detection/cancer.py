import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import pkg_resources

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

resource_package = __name__
resource_path = '/'.join(('datasets', 'cancer.csv'))  
template = pkg_resources.resource_stream(resource_package, resource_path)


dataset = pd.read_csv(template)
dataset = dataset.iloc[:, :-1]

X = dataset.drop(['id', 'diagnosis'], axis=1)
Y = dataset['diagnosis']

# (M = malignant, B = benign) 

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(X_train, Y_train)

classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_knn.fit(X_train, Y_train)

classifier_svc = SVC(kernel = 'linear', random_state = 0)
classifier_svc.fit(X_train, Y_train)

classifier_nb = GaussianNB()
classifier_nb.fit(X_train, Y_train)

classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_rf.fit(X_train, Y_train)

class cancer:
    def __init__(self, X_inp):
        """Initialize the class"""
        self.flag = 0
        if X_inp is not None:
            for i in X_inp:
                if len(i) != 30:
                    self.flag = 1
                    raise ValueError("Input array length must be 30 for all rows")
        
                if not all(isinstance(element, (int, float)) for element in i):
                    self.flag = 1
                    raise ValueError("All elements in the input array must be numeric")
        else:
            self.flag = 1
            raise ValueError("Input cannot be None")
        
        if self.flag == 0:
            self.X_inp = X_inp
        

    def LogisticRegression(self):
        if self.flag != 0:
            return 'Initialization failed'
        return list(classifier_lr.predict(self.X_inp))
    
    def KNearestNeighbours(self):
        if self.flag != 0:
            return 'Initialization failed'
        return list(classifier_knn.predict(self.X_inp)) 
    
    def SupportVectorClassifier(self):
        if self.flag != 0:
            return 'Initialization failed'
        return list(classifier_svc.predict(self.X_inp)) 
    
    def GNB(self):
        if self.flag != 0:
            return 'Initialization failed'
        
        return list(classifier_nb.predict(self.X_inp)) 
    
    def RandomForest(self):
        if self.flag != 0:
            return 'Initialization failed'
        return list(classifier_rf.predict(self.X_inp))