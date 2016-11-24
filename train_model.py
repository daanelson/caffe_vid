import caffe
import os
import numpy as np
import get_caffe_data

caffe.set_mode_gpu()
caffe_root = '/work/04035/dnelson8/maverick/caffe'
model_root = '/work/04035/dnelson8/maverick/vr_project/caffe_vid'
model_prototxt = os.path.join(model_root, 'train_net.prototxt')
model_solver = os.path.join(model_root, 'train_solver.prototxt')

label_root = '/work/04035/dnelson8/maverick/vr_project/dataset/ucfTrainTestlist'
h5_root = '/work/04035/dnelson8/maverick/vr_project/dataset/UCF-101-extract'
split = 1

net = caffe.Net(model_prototxt, caffe.TRAIN)

solver = caffe.SGDSolver(model_solver)

# manually load data for now b/c why not
counter = 0

train_list, test_list = get_caffe_data.get_train_test_lists(label_root, split)
print 'loading:'
TRUNCATE_FOR_TESTING = 300
train_data, train_label, test_data, test_label = get_caffe_data.get_train_test_data_labels(train_list[:300], test_list[:300], h5_root)

# code to get train_data in the proper shape:
train_data = np.array(train_data)
train_data = np.ascontiguousarray(train_data[:,np.newaxis,:,:])

train_label = np.ascontiguousarray(train_label)

#TODO: will need to do this w/test data at some point

net.set_input_arrays(train_data, train_label)

# solving:
niter = 1000
batch_start = 0
batch_size = 100
batch_end = batch_size


for it in range(niter):
    # solver.net.blobs['data'].data[:,:,:,:] = train_data[batch_start:batch_end]
    # solver.net.blobs['label'].data[:] = train_label[batch_start:batch_end]
    solver.step(1)
    print 'stepped'

    # manual batching like an asshole
    # batch_start = batch_end
    # batch_end = (batch_end + 100) % TRUNCATE_FOR_TESTING
    # if batch_end < batch_size:
    #     batch_start = 0
    #     batch_end = batch_size
