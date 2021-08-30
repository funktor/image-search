import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pickle

class SearchModel:
    def __init__(self, full_model_path, img_shape=(128,128,3), epochs=5, learning_rate=0.001, 
                 steps_per_epoch=100, callbacks=None, optimizer=None):
        
        self.model_path = full_model_path
        self.img_shape = img_shape
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.callbacks = callbacks
        self.model = None
        self.emb_model = None
        self.optimizer = optimizer
        self.steps_per_epoch = steps_per_epoch
        
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
        
        self.model.compile(optimizer=self.optimizer,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])
    
    def fit(self, training_dataset):
        self.model.fit(training_dataset, 
                       steps_per_epoch=self.steps_per_epoch, 
                       epochs=self.epochs,
                       callbacks=self.callbacks)
    
    def predict(self, img_array):
        if len(img_array.shape) == 1:
            img_array = tf.expand_dims(img_array, 0)
            
        return self.emb_model.predict(img_array)
    
    def predict_on_batch(self, img_batch):
        return self.emb_model.predict_on_batch(img_batch)
    
    def save(self, model=None):
        if model is None:
            self.emb_model.save(self.model_path)
        else:
            model.save(self.model_path)
    
    def load(self):
        model = tf.keras.models.load_model(self.model_path)
        return model
