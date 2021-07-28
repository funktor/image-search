from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os, json
import tensorflow as tf
from sklearn.neighbors import BallTree
import pickle
import importlib
from model import SearchModel

prefix = '/opt/ml/'

input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def get_file_list(root_dir):
    file_list = []
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
    return file_list

def train():
    try:
        print('Getting hyperparameters...')
        with open(param_path, 'r') as tc:
            training_params = json.load(tc)
            
        img_size = (int(training_params['img_width']), int(training_params['img_height']))
        batch_size = int(training_params['batch_size'])
        img_shape = (int(training_params['img_width']), int(training_params['img_height']), 3)
        epochs = int(training_params['epochs'])
        learning_rate = float(training_params['lr'])
            
        print('Splitting training and validation...')
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(input_path, 
                                                                   validation_split=0.2,
                                                                   subset="training",
                                                                   seed=42,
                                                                   image_size=img_size,
                                                                   batch_size=batch_size)

        validn_ds = tf.keras.preprocessing.image_dataset_from_directory(input_path, 
                                                                    validation_split=0.2,
                                                                    subset="validation",
                                                                    seed=42,
                                                                    image_size=img_size,
                                                                    batch_size=batch_size)
        num_classes = len(train_ds.class_names)

        print('Setting prefetch...')
        AUTOTUNE = tf.data.AUTOTUNE

        train_ds_pf = train_ds.prefetch(buffer_size=AUTOTUNE)
        validn_ds_pf = validn_ds.prefetch(buffer_size=AUTOTUNE)

        print('Initializing model object...')
        
        model_obj = SearchModel(full_model_path=model_path, img_shape=img_shape, epochs=epochs, learning_rate=learning_rate)

        model_obj.init(num_classes)

        print('Training model...')
        model_obj.fit(train_ds_pf, validn_ds_pf)

        filenames = sorted(get_file_list(input_path))

        print('Computing embeddings...')
        embeddings = np.empty(shape=(0, 2048))

        all_dataset = tf.keras.preprocessing.image_dataset_from_directory(data_dir, shuffle=False, image_size=img_size, batch_size=128)

        for image_batch, _ in all_dataset.as_numpy_iterator():
            embeddings = np.concatenate((embeddings, model_obj.predict_on_batch(image_batch)), axis=0)

        print('Saving embeddings...')
        with open('model/embeddings.pkl', 'wb') as f:
            pickle.dump((embeddings, filenames), f, protocol=pickle.HIGHEST_PROTOCOL)

        print('Creating balltree...')
        tree = BallTree(embeddings, leaf_size=5)

        print('Saving balltree...')
        with open('model/ball_tree.pkl', 'wb') as f:
            pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print('Printing similar images...')
        dist, ind = tree.query(embeddings[2333:2334], k=5)

        plt.figure(figsize=(10, 10))

        i = 0
        for idx in ind[0]:
            ax = plt.subplot(1, len(ind[0]), i+1)
            file = filenames[idx]
            img = tf.keras.preprocessing.image.load_img(file, target_size=img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            plt.imshow(img_array / 255)
            plt.axis('off')
            i += 1

        plt.savefig(os.path.join(output_path, 'similar_images.png'))

        print('Completed !!!')
        
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
            
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)
    
if __name__=="__main__":
    train()
    sys.exit(0)