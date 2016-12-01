import numpy as np
import os
#Prevent caffe output
os.environ['GLOG_minloglevel'] = '2'
import caffe
import csv

# Set the correct paths
# Download the pretrained CNN from https://github.com/metalbubble/places365
mean_path = '/home/nick/local/models/caffe/places_cnn/places365CNN_mean.binaryproto'
labels_path = '/home/nick/local/models/caffe/places_cnn/categories_places365.txt'
model_path = '/home/nick/local/models/caffe/places_cnn/deploy_vgg16_places365.prototxt'
weights_path = '/home/nick/local/models/caffe/places_cnn/vgg16_places365.caffemodel'

class Places_CNN:

    def __init__(self):
        blob = caffe.proto.caffe_pb2.BlobProto()
        data = open(mean_path, 'rb').read()
        blob.ParseFromString(data)
        arr = np.array(caffe.io.blobproto_to_array(blob))
        self.mean = arr[0]
        print self.mean.mean(1).mean(1)
        # Load caffe model
        self.net = caffe.Net(model_path,  caffe.TEST, weights=weights_path)

        # Create a transformer
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_mean('data', self.mean.mean(1).mean(1))
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_channel_swap('data', (2, 1, 0))
        self.transformer.set_raw_scale('data', 255.0)

        # Load classes
        self.labels = np.loadtxt(labels_path, str, delimiter='\t')

        # Output variables
        self.out = None
        self.conv_features = None


    def process_batch(self, images):
        batch_size = len(images)
        self.net.blobs['data'].reshape(batch_size, 3, 224, 224)
        for i, image in enumerate(images):
            self.net.blobs['data'].data[i, :, :, :] = self.transformer.preprocess('data', image)

        self.out = self.net.forward()['prob']

        # Options for convolutional features
        self.conv_features = self.net.blobs['conv5_3'].data

    def get_conv_features(self):
        return self.conv_features.transpose((0, 2, 3, 1))

    def get_output(self):
        res_idx = []
        for i in range(self.out.shape[0]):
            res_idx.append(self.out[i, :].argsort()[::-1])
        return res_idx

    def get_features(self):
        return self.features


    def load_image(self, path):
        return caffe.io.load_image(path)
