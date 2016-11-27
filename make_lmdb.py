import caffe
import lmdb
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum
import os
import get_caffe_data


def write_to_lmdb(image_list, db_path, h5_root):

    map_size = 150000000000
    env = lmdb.Environment(db_path, map_size=map_size)
    with env.begin(write=True, buffers=True) as txn:
        for idx, image in enumerate(image_list):
            # 150 GB
            X = get_caffe_data.load_h5_file(os.path.join(h5_root, image[0]))
            y = image[1]
            datum = array_to_datum(X, y)
            str_id = '{:08}'.format(idx)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

        txn.commit()
    env.close()
    print " ".join(["Writing to", db_path, "done!"])    