import caffe
import lmdb
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum
import os
import get_caffe_data
import numpy as np


def write_to_lmdb(image_list, db_path, h5_root):

    map_size = 150000000000
    env = lmdb.Environment(db_path, map_size=map_size)
    with env.begin(write=True, buffers=True) as txn:
        for idx, image in enumerate(image_list):
            # 150 GB
            X = get_caffe_data.load_h5_file(os.path.join(h5_root, image[0]))
            X = X[np.newaxis,:,:]
            y = int(image[1])
            datum = array_to_datum(X, y)
            str_id = '{:08}'.format(idx)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

    env.close()
    print " ".join(["Writing to", db_path, "done!"])



label_root = '/work/04035/dnelson8/maverick/vr_project/dataset/ucfTrainTestlist'
h5_root = '/work/04035/dnelson8/maverick/vr_project/dataset/UCF-101-extract'
split = 3
db_root = '/work/04035/dnelson8/maverick/vr_project/dataset/lmdb/' + str(split)

train_list, test_list = get_caffe_data.get_train_test_lists(label_root, split)
write_to_lmdb(train_list, os.path.join(db_root, 'train'), h5_root)


