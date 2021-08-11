#!/usr/bin/env python

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os, json, sys, math, argparse
import tensorflow as tf
from sklearn.neighbors import BallTree
import pickle, glob
import importlib
from model import SearchModel
from sagemaker_tensorflow import PipeModeDataset

prefix = '/opt/ml/'

input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

image_feature_description = {
    'image_name': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string)
}

extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def get_file_list(root_dir):
    file_list = []
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
    return file_list

def _dataset_parser(example_proto):
    features = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.io.decode_raw(features["image_raw"], tf.uint8)
    image = tf.reshape(image, [128, 128, 3])
    image = tf.cast(image, tf.float32)
    
    label = tf.cast(features["label"], tf.int32)
    
    return image, label

def _dataset_imagename_parser(example_proto):
    features = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.io.decode_raw(features["image_raw"], tf.uint8)
    image = tf.reshape(image, [128, 128, 3])
    image = tf.cast(image, tf.float32)
    
    return image, features['image_name']

def get_dataset_from_tfrecords(channel_name, batch_size, epochs):
    if 'RUNTIME_ENV' not in os.environ or os.environ['RUNTIME_ENV'] == 'local':
        dataset = tf.data.TFRecordDataset([f for f in glob.glob(os.path.join(input_path, channel_name, "*.tfrecords"))])
    else:
        dataset = PipeModeDataset(channel=channel_name, record_format="TFRecord")
    
    dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(10)
    dataset = dataset.map(_dataset_parser, num_parallel_calls=10)

    if channel_name == "train":
        buffer_size = 3 * batch_size
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    
    return dataset

def get_image_names_from_tfrecords(channel_name, batch_size, epochs):
    if 'RUNTIME_ENV' not in os.environ or os.environ['RUNTIME_ENV'] == 'local':
        dataset = tf.data.TFRecordDataset([f for f in glob.glob(os.path.join(input_path, channel_name, "*.tfrecords"))])
    else:
        dataset = PipeModeDataset(channel=channel_name, record_format="TFRecord")
    
    dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(10)
    dataset = dataset.map(_dataset_imagename_parser, num_parallel_calls=10)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    
    return dataset

def train():
    print('Getting hyperparameters...')
    with open(param_path, 'r') as tc:
        training_params = json.load(tc)

    img_size = (int(training_params['img_width']), int(training_params['img_height']))
    batch_size = int(training_params['batch_size'])
    img_shape = (int(training_params['img_width']), int(training_params['img_height']), 3)
    epochs = int(training_params['epochs'])
    learning_rate = float(training_params['lr'])
    run_id = int(training_params['run_id'])
    is_parallel = training_params['is_parallel']
    
    mpi = None
    
    if is_parallel.lower() == "true":
        import horovod.tensorflow.keras as hvd

        mpi = True
        hvd.init()

        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
            
    else:
        hvd = None
        
    if hvd is not None:
        channel_name = str(hvd.local_rank())
    else:
        channel_name = "0"

    print('Getting training datasets...')
    complete_dataset = get_dataset_from_tfrecords(channel_name, batch_size, 1)
    
    num_classes = len(set(np.concatenate([y for x, y in complete_dataset], axis=0).tolist()))
    print(num_classes)

    print('Initializing model object...')
    model_obj = SearchModel(full_model_path=os.path.join(model_path, 'image_search_model/'+str(run_id)), 
                            img_shape=img_shape, epochs=epochs, learning_rate=learning_rate, mpi=mpi, hvd=hvd)
    model_obj.init(num_classes=num_classes)

    print('Training model...')
    model_obj.fit(complete_dataset)

    print('Computing embeddings...')
    embeddings = np.empty(shape=(0, 2048))
    complete_dataset = get_image_names_from_tfrecords(channel_name, batch_size, 1)
    
    filenames = []
    
    for img_array_batch, image_names_batch in complete_dataset:
        embeddings = np.concatenate((embeddings, model_obj.predict_on_batch(img_array_batch)), axis=0)
        for file in image_names_batch:
            filenames.append(file.numpy().decode('utf-8'))
       
    print(embeddings.shape)
    
    print(filenames[:10])
    
    print('Saving image names...')
    with open(os.path.join(model_path, 'filenames.pkl'), 'wb') as f:
        pickle.dump(filenames, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Creating balltree...')
    tree = BallTree(embeddings, leaf_size=5)

    print('Saving balltree...')
    with open(os.path.join(model_path, 'ball_tree.pkl'), 'wb') as f:
        pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Completed !!!')
    
if __name__=="__main__":
    train()
    sys.exit(0)