from Variable import *
import os
import numpy as np
import joblib
import random

def transform_type(input_file, output_file):
    with open(output_file, "w") as text_file:
        with open(input_file) as f:
            lines = f.readlines()
            for line in lines:
                columns = line.split(',')
                for raw_type in category:
                    flag = False
                    if raw_type == columns[-1].replace("\n", ""):
                        str = ','.join(columns[0:attr_list.index('type')])
                        text_file.write("%s,%d\n" % (str, category[raw_type]))
                        flag = True
                        break
                if not flag:
                    text_file.write(line)
                    print(line)


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def save_data(data_file, out_file_x, out_file_y, save_dic=False):
    with open(data_file) as f:
        lines = f.readlines()
        data = []
        y = []
        for line in lines:
            columns = line.split(',')
            dic = {}
            x = []
            for attr in attr_list:
                element = columns[attr_list.index(attr)]
                if element.isdigit():
                    element = int(element)
                elif isfloat(element):
                    element = float(element)
                if save_dic:
                    if attr != 'type':
                        dic[attr] = element
                    else:
                        y.append(element)
                else:
                    x.append(element)
            if save_dic:
                data.append(dic)
            else:
                data.append(x)
        data = np.asarray(data)
        print(data.shape)
        if save_dic:
            x = data
            y = np.asarray(y)
        else:
            x = data[:, :-1]
            y = data[:, -1]

        y = y.astype(float).astype(int)
        np.save(out_file_x, x)
        np.save(out_file_y, y)

def cross_validation_split(dataset, target, factor=0.1):
    val_index = random.sample(range(0, len(target) - 1), int(len(target) * factor))
    training_index = list(set(range(0, len(target) - 1)) - set(val_index))

    training_set = dataset[training_index]
    training_target = target[training_index]

    val_set = dataset[val_index]
    val_target = target[val_index]

    print("\n")
    print("training_set: " + str(training_set.shape))
    print("training_target: " + str(training_target.shape))

    print("val_set: " + str(val_set.shape))
    print("val_target: " + str(val_target.shape))
    print("\n")

    return training_set, training_target, val_set, val_target


