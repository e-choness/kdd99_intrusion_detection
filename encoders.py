from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
import numpy as np



def label_encoding(dataset):
    le_1 = preprocessing.LabelEncoder()
    le_2 = preprocessing.LabelEncoder()
    le_3 = preprocessing.LabelEncoder()

    le_1.fit(np.unique(dataset[:, 1]))
    le_2.fit(np.unique(dataset[:, 2]))
    le_3.fit(np.unique(dataset[:, 3]))

    dataset[:, 1] = le_1.transform(dataset[:, 1])
    dataset[:, 2] = le_2.transform(dataset[:, 2])
    dataset[:, 3] = le_3.transform(dataset[:, 3])
    return dataset


def one_hot_encoding(dataset):
        vec = DictVectorizer()
        dataset = vec.fit_transform(dataset).toarray()
        print(dataset.shape)
        return dataset
