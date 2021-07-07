from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from Variable import *

def fetch_dataset(dataset, nums_train, encoder, feature_select, val_split, pca_flag=False):
    x, y = dataset
    x = encoder(x)
    x_train = x[:nums_train]
    x_test = x[nums_train:]
    y_train = y[:nums_train]
    y_test = y[nums_train:]
    print(x_train.shape, x_test.shape, x.shape)

    if pca_flag == True:
        pca = PCA(n_components=20)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
        print(x_train.shape)
        scaler = StandardScaler()
        scaler.fit(x_train)
        print(pca.explained_variance_ratio_)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
    if feature_select:
        x_featured, feature_index = feature_select(x_train, y_train)
        x_train, y_train, x_val, y_val = val_split(x_featured, y_train)
        x_test = x_test[:, feature_index]
        print(x_train.shape, x_test.shape, x_val.shape, y_val.shape)
        return (x_train, y_train), (x_val, y_val), (x_test, y_test), feature_index
    else:
        x_train, y_train, x_val, y_val = val_split(x_train, y_train)
        print(x_train.shape, x_test.shape, x_val.shape, y_val.shape)
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)