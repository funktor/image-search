from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, math, random
import argparse
from PIL import Image
from sklearn.model_selection import train_test_split

extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_files_list(root_dir, validation_split=0.2):
    train_x, valid_x = [], []
    train_y, valid_y = [], []
    
    all_labels = set()
    
    for root, directories, filenames in os.walk(root_dir):
        for d in directories:
            files = [os.path.join(root, d, filename) for filename in os.listdir(os.path.join(root, d)) if any(ext in filename for ext in extensions)]
            labels = [d]*len(files)
            
            t_x, v_x, t_y, v_y = train_test_split(files, labels, test_size=validation_split, random_state=42)
            train_x += t_x
            valid_x += v_x
            train_y += t_y
            valid_y += v_y
            
            all_labels.add(d)
     
    u = list(set(all_labels))
    h = {u[i]:i for i in range(len(u))}
    
    train_dataset = list(zip(train_x, [h[x] for x in train_y]))
    valid_dataset = list(zip(valid_x, [h[x] for x in valid_y]))
    
    random.shuffle(train_dataset)
    random.shuffle(valid_dataset)
    
    train_x, train_y = zip(*train_dataset)
    valid_x, valid_y = zip(*valid_dataset)
    
    return train_x, valid_x, train_y, valid_y

def image_example(image_string, label, image_name):
    feature = {
        'image_name': _bytes_feature(image_name),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def convert_to_tfrecords(filenames, labels, tfrecords_file):
    with tf.io.TFRecordWriter(tfrecords_file) as writer:
        for i in range(len(filenames)):
            image_name = filenames[i].split('/')
            image_name = bytes(os.path.join(image_name[-2], image_name[-1]), 'utf-8')
            img = Image.open(filenames[i])
            image_string = img.resize((128, 128), Image.ANTIALIAS).convert('RGB').tobytes()
            tf_example = image_example(image_string, labels[i], image_name)
            writer.write(tf_example.SerializeToString())
            
def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, image_feature_description)

class TFRecordsData:
    def __init__(self, input_path, train_output_path, valid_output_path, full_data_output_path, validation_split=0.2):
        self.input_path = input_path
        self.train_output_path = train_output_path
        self.valid_output_path = valid_output_path
        self.full_data_output_path = full_data_output_path
        self.validation_split = validation_split
    
    def convert_to_tfrecords(self):
        train_x, valid_x, train_y, valid_y = get_files_list(self.input_path, self.validation_split)
        
        convert_to_tfrecords(train_x, train_y, self.train_output_path)
        convert_to_tfrecords(valid_x, valid_y, self.valid_output_path)
        convert_to_tfrecords(train_x+valid_x, train_y+valid_y, self.full_data_output_path)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args, _ = parser.parse_known_args()

    input_folder = args.input
    output_folder = args.output

    tf_record_generator = TFRecordsData(input_path=input_folder,
                                        train_output_path=os.path.join(output_folder, 'train.tfrecords'), 
                                        valid_output_path=os.path.join(output_folder, 'valid.tfrecords'), 
                                        full_data_output_path=os.path.join(output_folder, 'complete.tfrecords'), 
                                        validation_split=0.2)

    print('GENERATING TF RECORD FILES...')
    tf_record_generator.convert_to_tfrecords()

    print('FINISHED')