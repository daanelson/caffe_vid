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


# Script!
label_root = '/Volumes/usb_storage/ucfTrainTestlist'
h5_root = '/Volumes/usb_storage/UCF-101-extract'

scores = []

for split in reversed(range(1,4)):
    # global (I know, I know)
    counter = 0

    train_list, test_list = get_train_test_lists(label_root, split)
    print 'loading:'
    train_data, train_label, test_data, test_label = get_train_test_data_labels(train_list, test_list, h5_root)

    print 'training:'
    clf = svm.SVC()
    clf.fit(train_data, train_label)
    scores.append(clf.score(test_data, test_label))

    print 'Split: ', split, ' Scores: ', scores

print 'Mean Score: ', np.mean(scores)


