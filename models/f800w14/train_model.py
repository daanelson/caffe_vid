import caffe
import os
import numpy as np
import gc
import sys

#this may switch to CPU, unsure
#caffe.set_device(0)
caffe.set_mode_gpu()
caffe_root = '/work/04035/dnelson8/maverick/caffe'
model_root = '/work/04035/dnelson8/maverick/vr_project/caffe_vid/models/f800w14'
model_prototxt = os.path.join(model_root, 'train_net{0}.prototxt'.format(sys.argv[1]))
model_solver = os.path.join(model_root, 'train_solver{0}.prototxt'.format(sys.argv[1]))

label_root = '/work/04035/dnelson8/maverick/vr_project/dataset/ucfTrainTestlist'
h5_root = '/work/04035/dnelson8/maverick/vr_project/dataset/UCF-101-extract'
split = int(sys.argv[1])

solver = caffe.SGDSolver(model_solver)

# manually load data for now b/c why not
counter = 0

'''train_list, test_list = get_caffe_data.get_train_test_lists(label_root, split)
print 'loading:'
TRUNCATE_FOR_TESTING = 300
train_data, train_label, test_data, test_label = get_caffe_data.get_train_test_data_labels(train_list[:9500], test_list[:3600], h5_root)

print 'loaded:'
# code to get train_data in the proper shape:
train_data = np.array(train_data, dtype=np.float32)
train_data = np.ascontiguousarray(train_data[:,np.newaxis,:,:])

train_label = np.ascontiguousarray(train_label, dtype=np.float32)
print 'train manipulated:'
# Need copy of net for labels to be properly set as per: https://github.com/BVLC/caffe/issues/4131
net_1 = solver.net
net_1.set_input_arrays(train_data, train_label)
gc.collect()
print 'test manipulating:'
# as per https://github.com/BVLC/caffe/pull/1196
test_data = np.array(test_data, dtype=np.float32)
print 'array'
test_data = np.ascontiguousarray(test_data[:,np.newaxis,:,:])
print 'contiguous'
test_label = np.ascontiguousarray(test_label, dtype=np.float32)
gc.collect()

print 'set'
net_2 = solver.test_nets[0]
net_2.set_input_arrays(test_data, test_label)
'''
# solving:
solver.solve()

