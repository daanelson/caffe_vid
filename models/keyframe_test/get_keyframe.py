import caffe
import numpy as np
import os
import h5py
import sys
import get_caffe_data
from collections import Counter
import pickle


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

#REMOVE TO WORK:

cur_start = 0
cur_finish = 10
increment = 10
frames = []

net.blobs['data'].reshape(10,1,150,4096)

while cur_finish < len(test_list):
    cur_list = test_list[cur_start:cur_finish]
    test_data = [get_caffe_data.load_h5_file(os.path.join(h5_root, val[0])) for val in cur_list]
    test_data = np.array(test_data, dtype=np.float32)
    test_data = np.ascontiguousarray(test_data[:,np.newaxis,:,:])
    net.blobs['data'].reshape(10,1,150,4096)
    net.blobs['data'].data[:,:,:,:] = test_data
    net.forward()
    argmax_vals = net.blobs['conv1'].data.argmax(2)
    for val in range(10):
        cur_counter = Counter(argmax_vals[val,:,0])
        best_frame = cur_counter.most_common(1)[0][0]
        frames.append(best_frame)

    cur_start += increment
    cur_finish += increment

final_list = test_list[cur_start:]
test_data = [get_caffe_data.load_h5_file(os.path.join(h5_root, val[0])) for val in final_list]
test_data = np.array(test_data, dtype=np.float32)
test_data = np.ascontiguousarray(test_data[:,np.newaxis,:,:])

batch_size = len(test_data)
net.blobs['data'].reshape(batch_size,1,150,4096)
net.blobs['data'].data[:,:,:,:] = test_data
net.forward()
argmax_vals = net.blobs['conv1'].data.argmax(2)
for val in range(len(final_list)):
    cur_counter = Counter(argmax_vals[val,:,0])
    best_frame = cur_counter.most_common(1)[0][0]
    frames.append(best_frame)

everything = zip(test_list, frames)

with open('bestframes', 'wb') as f:
    pickle.dump(everything, f)
