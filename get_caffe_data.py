import os
import h5py
import numpy as np
import pickle
import pdb
from sklearn import svm


def get_train_test_lists(label_root, split):
    # hey hey it's reproducibility!
    np.random.seed(12345)

    #Training Data
    with open(os.path.join(label_root,'trainlist0' + str(split) + '.txt'), 'rb') as f:
        wrong_path = [val.strip().split(' ') for val in f.readlines()]
        train_list = np.random.shuffle([[x[0].split('.')[0] + '.h5', x[1]] for x in wrong_path])

    # Class dict to provide labels for test data
    with open(os.path.join(label_root,'classInd.txt'), 'rb') as f:
        class_list = [val.strip().split(' ') for val in f.readlines()]
        class_to_label = {val[1]:val[0] for val in class_list}

    # Test Data
    with open(os.path.join(label_root, 'testlist0'+str(split) + '.txt'), 'rb') as f:
        test_list = np.random.shuffle([[val.strip().split('.')[0] + '.h5', class_to_label[val.split('/')[0]]] for val in f.readlines()])

    return train_list, test_list


# returning only one h5 file at a time,
def load_h5_file(file_path):
    global counter
    counter += 1
    if counter % 100 == 0:
        print counter
    try:
        with h5py.File(file_path, 'r') as f:
            data = f.get('data')
            np_data = np.array(data)
            f.close()
        return np_data
    except IOError:
        print file_path
        return np.zeros(300)


def get_train_test_data_labels(train_list, test_list, h5_root):
    # Only load that data once
    train_data = [load_h5_file(os.path.join(h5_root, val[0])) for val in train_list]
    test_data = [load_h5_file(os.path.join(h5_root, val[0])) for val in test_list]

    train_label = [val[1] for val in train_list]
    test_label = [val[1] for val in test_list]
    return train_data, train_label, test_data, test_label


