import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pickle

class SearchModel:
    def __init__(self, full_model_path, img_shape=(128,128,3), epochs=5, learning_rate=0.001, mpi=None, hvd=None):
        self.model_path = full_model_path
        self.img_shape = img_shape
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.callbacks = []
        self.model = None
        self.emb_model = None
        self.mpi = mpi
        self.hvd = hvd
        self.opt = None
        
    def init(self, num_classes):
        data_augmentation = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
          tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])
        
        base_model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape = self.img_shape)
        base_model.trainable = False
        
        inputs = tf.keras.Input(shape=self.img_shape)
        x = data_augmentation(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        self.model = tf.keras.Model(inputs, outputs)
        self.emb_model = tf.keras.Model(self.model.input, self.model.layers[-2].output)
        
        if self.mpi:
            self.learning_rate *= self.hvd.size()
            
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        print(self.model_path)
        
        if self.mpi:
            print(self.hvd.rank())
            self.opt = self.hvd.DistributedOptimizer(self.opt)
            
            self.callbacks.append(self.hvd.callbacks.BroadcastGlobalVariablesCallback(0))
            self.callbacks.append(self.hvd.callbacks.MetricAverageCallback())
            self.callbacks.append(self.hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))
            
            if self.hvd.rank() == 0:
                self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path, 
                                                                 save_weights_only=False,
                                                                 verbose=1))
        else:
            self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path, 
                                                                 save_weights_only=False,
                                                                 verbose=1))
        
        self.model.compile(optimizer=self.opt,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])
    
    def fit(self, training_dataset, validation_dataset=None):
        self.model.fit(training_dataset, epochs=self.epochs, validation_data=validation_dataset, callbacks=self.callbacks)
        
        if self.mpi:
            if self.hvd.rank() == 0:
                self.emb_model.save(self.model_path)
        else:
            self.emb_model.save(self.model_path)
    
    def predict(self, img_array):
        if len(img_array.shape) == 1:
            img_array = tf.expand_dims(img_array, 0)
            
        return self.emb_model.predict(img_array)
    
    def predict_on_batch(self, img_batch):
        return self.emb_model.predict_on_batch(img_batch)
    
    def save(self):
        if self.mpi:
            if self.hvd.rank() == 0:
                self.model.save(self.model_path)
        else:
            self.model.save(self.model_path)
    
    def load(self):
        self.model = tf.keras.models.load_model(self.model_path)
        self.emb_model = tf.keras.Model(self.model.input, self.model.layers[-2].output)