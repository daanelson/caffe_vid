import caffe
import numpy as np
import os
import h5py
import sys
import get_caffe_data


caffe.set_mode_gpu()
caffe_root = '/work/04035/dnelson8/maverick/caffe'
label_root = '/work/04035/dnelson8/maverick/vr_project/dataset/ucfTrainTestlist'
h5_root = '/work/04035/dnelson8/maverick/vr_project/dataset/UCF-101-extract'


model_root = '/work/04035/dnelson8/maverick/vr_project/caffe_vid/models/f800w7'
model_prototxt = os.path.join(model_root, 'train_net.prototxt')
model_weights = os.path.join(model_root, 'model/dannet_iter_2000.caffemodel')

work_dir = '/work/04035/dnelson8/maverick/'
top_image_dir = os.path.join(work_dir, 'vr_project/dataset/UCF-101-images')
feat_extract_dir = os.path.join(work_dir, 'vr_project/dataset/UCF-101-extract')

net = caffe.Net(model_prototxt, model_weights, caffe.TEST)

# define preprocessing
split = 1
_, test_list = get_caffe_data.get_train_test_lists(label_root, split)

cur_start = 0
cur_finish = 10
increment = 10

while cur_finish < len(test_list):
    test_data = [get_caffe_data.load_h5_file(os.path.join(h5_root, val[0])) for val in test_list[cur_start:cur_finish]]
    test_data = np.array(test_data, dtype=np.float32)
    test_data = np.ascontiguousarray(test_data[:,np.newaxis,:,:])

    net.blobs['data'].data = test_data
    net.forward()
    pdb.set_trace()
    net.blobs['conv1'].data.argmax()

    cur_start += increment
    cur_finish += increment