from sklearn import tree
from sklearn.neural_network import MLPClassifier
import joblib
import os
import collections
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class clf_models():
    def decistionTree(self):
        clf = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=30, class_weight="balanced")
        return clf
    def MLP(self):
        clf = MLPClassifier(solver='adam', alpha=1e-5,
                            hidden_layer_sizes=(10, 6), random_state=1)
        return clf
    def train(self, clf, x_train, y_train, model_dir=None, model_name=None):
        clf = clf.fit(x_train, y_train)
        if model_name != None and model_dir != None:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            joblib.dump(clf, os.path.join(model_dir, model_name))
    def evaluate(self, clf, x_test, y_test, model_dir=None, model_name=None, mode='validation'):
        if model_name != None and model_dir != None:
            clf = joblib.load(os.path.join(model_dir, model_name))
        y_predict = clf.predict(x_test)
        if mode == "validation":
            print("y_validation: " + str(collections.Counter(y_test)))
        elif mode == "test":
            print("y_test: " + str(collections.Counter(y_test)))
        print("y_predict: " + str(collections.Counter(y_predict)))
        print("\n")
        print(confusion_matrix(y_test, y_predict, labels=[0, 1, 2, 3, 4]))
        print("\n")
        print(classification_report(y_test, y_predict))
        print("\n")
