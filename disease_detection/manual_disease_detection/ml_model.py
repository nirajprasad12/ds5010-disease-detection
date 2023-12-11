from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    # Train Random Forest classifier
    clf_rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf_rf.fit(X_train, y_train)
    return clf_rf
    
def train_logistic_regression(X_train, y_train, random_state = 0):
    classifier_lr = LogisticRegression(random_state = random_state)
    classifier_lr.fit(X_train, y_train)
    return classifier_lr

def train_k_nearest_neighbours(X_train, y_train, n_neighbors = 5, metric = 'minkowski', p = 2):
    classifier_knn = KNeighborsClassifier(n_neighbors = n_neighbors, metric = metric, p = p)
    classifier_knn.fit(X_train, y_train)
    return classifier_knn

def evaluate_model(clf, X_test, y_test, req_result = 0):
    # Make predictions
    y_pred = clf.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    if req_result == 0:
        return y_pred, accuracy, report
    elif req_result == 1:
        return y_pred
    elif req_result == 2:
        return accuracy
    elif req_result == 3:
        return report
    else:
        print('Wrong input flag chosen - refer documentation')

def plot_confusion_matrix(y_true, y_pred):
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()