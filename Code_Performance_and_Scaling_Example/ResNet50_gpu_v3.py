##### Import relevant modules #####

import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np
import sys
import pickle as cPickle
import os

def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
                d = d_decoded
    data = d['data']
    labels = d[label_key]
    
    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels




def load_cifar(batch_size, path='./data/cifar-10-batches-py'):

    train_images = np.empty((50000, 3, 32, 32), dtype='uint8')
    train_labels = np.empty((50000,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (train_images[(i - 1) * 10000: i * 10000, :, :, :],
            train_labels[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    test_images, test_labels = load_batch(fpath)


                                                                                                
    def preprocess_fn(image, label):

        label = tf.cast(label, tf.int32)

        if K.image_data_format() == 'channels_last':
            image = tf.transpose(tf.reshape(image, [3, 32, 32]), [1, 2, 0])
        image = tf.image.resize(image, [128,128])

        # Normalize data.
        image = tf.math.divide(tf.cast(image, tf.float32), 255)

        return image, label
                                                                                                                                                                            
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    train_ds = train_ds.map(preprocess_fn, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().repeat(1)
    test_ds = test_ds.map(preprocess_fn, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat()
                                                                                                                                                                                
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)                                                                                               
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)                                                                                                 
    
    return train_ds, test_ds

import tensorflow as tf 
from tensorflow.keras import layers 
from tensorflow.keras import models

import numpy as np
import argparse
import time
import sys
import argparse

start = time.perf_counter()

ap = argparse.ArgumentParser()
ap.add_argument("-n","--ngpus",type=int,default=1,help="number of GPUs to use")
ap.add_argument("-b","--batch",type=int,default=256,help="batch size")
args = vars(ap.parse_args())

n_gpus = args["ngpus"]
batch_size = args["batch"]
epochs = 10

train_ds, test_ds = load_cifar(batch_size)

device_type = 'GPU'
devices = tf.config.experimental.list_physical_devices(
                  device_type)
devices_names = [d.name.split('e:')[1] for d in devices]

strategy = tf.distribute.MirroredStrategy(devices=devices_names[:n_gpus])

with strategy.scope():
     model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=True, weights=None,
             input_shape=(128, 128, 3), classes=10)

     opt = tf.keras.optimizers.SGD(0.01*n_gpus)
     model.compile(loss='sparse_categorical_crossentropy', 
                   optimizer=opt, metrics=['accuracy'])
model.fit(train_ds, epochs=epochs, verbose=2)

elapsed = time.perf_counter() - start
print('Elapse %.3f seconds.' % elapsed)