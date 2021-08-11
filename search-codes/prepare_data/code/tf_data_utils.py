from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, math, random
import argparse
from PIL import Image
from sklearn.model_selection import train_test_split
import multiprocessing
from multiprocessing import Process

extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_files_list(root_dir):
    train_x, train_y = [], []
    
    all_labels = set()
    
    for root, directories, filenames in os.walk(root_dir):
        for d in directories:
            files = [os.path.join(root, d, filename) for filename in os.listdir(os.path.join(root, d)) if any(ext in filename for ext in extensions)]
            labels = [d]*len(files)
            train_x += files
            train_y += labels
            all_labels.add(d)
     
    u = list(set(all_labels))
    h = {u[i]:i for i in range(len(u))}
    
    train_dataset = list(zip(train_x, [h[x] for x in train_y]))
    
    random.shuffle(train_dataset)
    
    train_x, train_y = zip(*train_dataset)
    
    return train_x, train_y

def image_example(image_string, label, image_name):
    feature = {
        'image_name': _bytes_feature(image_name),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def convert_to_tfrecord(filenames, labels, tfrecords_file):
    with tf.io.TFRecordWriter(tfrecords_file) as writer:
        for i in range(len(filenames)):
            image_name = filenames[i].split('/')
            image_name = bytes(os.path.join(image_name[-2], image_name[-1]), 'utf-8')
            img = Image.open(filenames[i])
            image_string = img.resize((128, 128), Image.ANTIALIAS).convert('RGB').tobytes()
            tf_example = image_example(image_string, labels[i], image_name)
            writer.write(tf_example.SerializeToString())

def convert_to_tfrecords(filenames, labels, tfrecords_dir, n_instances, n_gpus):
    num_processes = n_instances*n_gpus
    batch_size = int(math.ceil(len(filenames)/num_processes))
    
    for j in range(num_processes):
        folder_index = int(j/n_instances)
        file_index = j % n_instances
        
        directory = os.path.join(tfrecords_dir, str(folder_index))
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        f = os.path.join(directory, "train_" + str(file_index) + ".tfrecords")
        p = Process(target=convert_to_tfrecord, args=(filenames[batch_size*j:min(batch_size*(j+1), len(filenames))], 
                                                      labels[batch_size*j:min(batch_size*(j+1), len(filenames))], 
                                                      f))
        p.start()
    
    p.join()
            
def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, image_feature_description)

class TFRecordsData:
    def __init__(self, input_path, out_dir, number_instances, num_gpus_per_instance):
        self.input_path = input_path
        self.out_dir = out_dir
        self.number_instances = number_instances
        self.num_gpus_per_instance = num_gpus_per_instance
    
    def convert_to_tfrecords(self):
        train_x, train_y = get_files_list(self.input_path)
        
        convert_to_tfrecords(train_x, train_y, 
                             self.out_dir, 
                             self.number_instances, 
                             self.num_gpus_per_instance)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--number_instances", type=int, default=1)
    parser.add_argument("--num_gpus_per_instance", type=int, default=1)
    args, _ = parser.parse_known_args()

    num_gpus_per_instance = args.num_gpus_per_instance
    input_folder = args.input
    output_folder = args.output
    number_instances = args.number_instances

    tf_record_generator = TFRecordsData(input_folder,
                                        output_folder, 
                                        number_instances, 
                                        num_gpus_per_instance)

    print('GENERATING TF RECORD FILES...')
    tf_record_generator.convert_to_tfrecords()

    print('FINISHED')