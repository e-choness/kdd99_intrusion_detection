from preprocess_data import save_data, cross_validation_split, transform_type
from clf_models import clf_models
from fetch_dataset import fetch_dataset
from feature_select_methods import tree_based_selection, threshold_feature_selection
from encoders import label_encoding, one_hot_encoding
import numpy as np


train_file = "data/kddcup.data_10_percent.txt"
test_file = 'data/corrected.txt'
transform_type("raw/kddcup.data_10_percent.txt", train_file)
transform_type("raw/corrected.txt", test_file)

save_data(train_file, './data/x_train_dic.npy', './data/y_train_dic.npy', save_dic=True)
save_data(test_file, './data/x_test_dic.npy', './data/y_test_dic.npy', save_dic=True)

save_data(train_file, './data/x_train.npy', './data/y_train.npy')
save_data(test_file, './data/x_test.npy', './data/y_test.npy')

CLF = clf_models()

x_train = np.load('./data/x_train.npy')
x_test = np.load('./data/x_test.npy')
y_train = np.load('./data/y_train.npy')
y_test = np.load('./data/y_test.npy')
x = np.concatenate((x_train, x_test))
print(y_train.shape, y_test.shape)
y = np.concatenate((y_train, y_test))
print(x_train.shape, x_test.shape, x.shape, y.shape)
nums_train = len(x_train)
dataset = (x, y)

# Decision Tree
decision_tree = CLF.decistionTree()
(x_train, y_train), (x_val, y_val), (x_test, y_test), feature_index = fetch_dataset(
                    dataset,
                    nums_train,
                    encoder=label_encoding,
                    feature_select=tree_based_selection,
                    val_split=cross_validation_split,
                    pca_flag=False
                )
CLF.train(decision_tree, x_train, y_train, model_dir='./output', model_name='CART.pkl')
CLF.evaluate(decision_tree, x_val, y_val,  model_dir='./output', model_name='CART.pkl')
CLF.evaluate(decision_tree, x_test, y_test,  model_dir='./output', model_name='CART.pkl', mode='test')


# MLP
x_train = np.load('./data/x_train_dic.npy', allow_pickle=True)
x_test = np.load('./data/x_test_dic.npy', allow_pickle=True)
y_train = np.load('./data/y_train_dic.npy', allow_pickle=True)
y_test = np.load('./data/y_test_dic.npy', allow_pickle=True)
x = np.concatenate((x_train, x_test))
print(y_train.shape, y_test.shape)
y = np.concatenate((y_train, y_test))
print(x_train.shape, x_test.shape, x.shape, y.shape)
nums_train = len(x_train)
dataset = (x, y)
(x_train, y_train), (x_val, y_val), (x_test, y_test), feature_index = fetch_dataset(
                    dataset,
                    nums_train,
                    encoder=one_hot_encoding,
                    feature_select=threshold_feature_selection,
                    val_split=cross_validation_split,
                    pca_flag=True
                )
mlp = CLF.MLP()
CLF.train(mlp, x_train, y_train, model_dir='./output', model_name='MLP.pkl')
CLF.evaluate(mlp, x_val, y_val,  model_dir='./output', model_name='MLP.pkl')
CLF.evaluate(mlp, x_test, y_test,  model_dir='./output', model_name='MLP.pkl', mode='test')