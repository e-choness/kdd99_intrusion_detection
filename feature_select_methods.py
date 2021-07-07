from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def tree_based_selection(dataset, target, *args, **kwargs):
    clf = ExtraTreesClassifier()
    clf = clf.fit(dataset, target)
    print('clf feature importance ', clf.feature_importances_)
    model = SelectFromModel(clf, prefit=True)
    feature_set = model.transform(dataset)
    print(feature_set.shape, feature_set[0])
    feature_index = []
    for A_col in np.arange(dataset.shape[1]):
        for B_col in np.arange(feature_set.shape[1]):
            if (dataset[:, A_col] == feature_set[:, B_col]).all():
                print('feature_index selected ', A_col, B_col)
                feature_index.append(A_col)
    return feature_set, feature_index


def threshold_feature_selection(dataset, target, *args, **kwargs):
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    feature_set = sel.fit_transform(dataset)
    feature_index = []
    for A_col in np.arange(dataset.shape[1]):
        for B_col in np.arange(feature_set.shape[1]):
            if (dataset[:, A_col] == feature_set[:, B_col]).all():
                print('feature_index selected ', A_col, B_col)
                feature_index.append(A_col)
    return feature_set, feature_index
