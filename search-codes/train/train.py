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

num_examples_per_epoch=29780

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

def get_dataset_from_tfrecords(channel, channel_name, batch_size, epochs, mode):
    if mode == 'Pipe':
        dataset = PipeModeDataset(channel=channel_name, record_format="TFRecord")
    else:
        dataset = tf.data.TFRecordDataset([f for f in glob.glob(os.path.join(channel, "*.tfrecords"))])
    
    dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(10)
    dataset = dataset.map(_dataset_parser, num_parallel_calls=10)

    if channel_name == "train":
        buffer_size = 3 * batch_size
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    
    return dataset

def get_image_names_from_tfrecords(channel, channel_name, batch_size, epochs, mode):
    if mode == 'Pipe':
        dataset = PipeModeDataset(channel=channel_name, record_format="TFRecord")
    else:
        dataset = tf.data.TFRecordDataset([f for f in glob.glob(os.path.join(channel, "*.tfrecords"))])
    
    dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(10)
    dataset = dataset.map(_dataset_imagename_parser, num_parallel_calls=10)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    
    return dataset

def train(args):
    print('Getting hyperparameters...')

    img_size = (args.img_width, args.img_height)
    batch_size = args.batch_size
    img_shape = (args.img_width, args.img_height, 3)
    epochs = args.epochs
    learning_rate = args.lr
    run_id = args.run_id
    
    mpi = None
    
    if 'sagemaker_mpi_enabled' in args.fw_params and args.fw_params['sagemaker_mpi_enabled']:
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
        channel_name = "complete"

    print('Getting training datasets...')
    complete_dataset = get_dataset_from_tfrecords(args.training_env['channel_input_dirs'][channel_name], 
                                                  channel_name, 
                                                  batch_size, epochs, 
                                                  args.training_env['input_data_config'][channel_name]['TrainingInputMode'])
    
    num_classes = 256

    print('Initializing model object...')
    model_path = os.path.join(args.model_output_dir, 'image_search_model/'+str(run_id))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    callbacks = []

    if hvd:
        size = hvd.size()
        optimizer = hvd.DistributedOptimizer(optimizer)

        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())
        callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))

        if hvd.rank() == 0:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=model_path, 
                                                                save_weights_only=False,
                                                                verbose=1))
    else:
        size = 1
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=model_path, 
                                                            save_weights_only=False,
                                                            verbose=1))
    
    g = ((num_examples_per_epoch // batch_size) // size)
    print(g*epochs)
    print('Initializing class...')
    model_obj = SearchModel(full_model_path=model_path, 
                            img_shape=img_shape, 
                            epochs=epochs, 
                            learning_rate=learning_rate,
                            steps_per_epoch=((num_examples_per_epoch // batch_size) // size),
                            callbacks=callbacks, 
                            optimizer=optimizer)
    
    print('Init...')
    model_obj.init(num_classes=num_classes)

    print('Training model...')
    model_obj.fit(complete_dataset)
    
    print('Saving model...')
    if hvd is None or hvd.rank() == 0:
        model_obj.save()

    print('Computing embeddings...')
    if hvd is None or hvd.rank() == 0:
        embeddings = np.empty(shape=(0, 2048))
        filenames = []
        
        channel_name = 'complete'
        ds = get_image_names_from_tfrecords(args.training_env['channel_input_dirs'][channel_name], 
                                            channel_name, batch_size, 1, 
                                            args.training_env['input_data_config'][channel_name]['TrainingInputMode'])

        for img_array_batch, image_names_batch in ds:
            embeddings = np.concatenate((embeddings, model_obj.predict_on_batch(img_array_batch)), axis=0)
            for file in image_names_batch:
                filenames.append(file.numpy().decode('utf-8'))

        print(embeddings.shape)

        print(filenames[:10])
        
        print('Saving image names...')
        with open(os.path.join(args.model_output_dir, 'filenames.pkl'), 'wb') as f:
            pickle.dump(filenames, f, protocol=pickle.HIGHEST_PROTOCOL)

        print('Creating balltree...')
        tree = BallTree(embeddings, leaf_size=5)

        print('Saving balltree...')
        with open(os.path.join(args.model_output_dir, 'ball_tree.pkl'), 'wb') as f:
            pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Completed !!!')
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_dir',type=str,required=True,help='The directory where the model will be stored.')
    parser.add_argument('--model_output_dir',type=str,default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--img_height',type=int,default=128)
    parser.add_argument('--img_width',type=int,default=128)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--run_id',type=int,default=1)
    parser.add_argument('--fw-params',type=json.loads,default=os.environ.get('SM_FRAMEWORK_PARAMS'))
    parser.add_argument('--training-env',type=json.loads,default=os.environ.get('SM_TRAINING_ENV'))
    
    args = parser.parse_args()
    
    train(args)
    sys.exit(0)