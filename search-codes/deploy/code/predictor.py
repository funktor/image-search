from __future__ import print_function

import matplotlib.pyplot as plt
import io
import json
import os
import pickle
import signal
import sys, glob, heapq
import traceback

import flask
import tensorflow as tf
from io import BytesIO

import base64
import numpy as np
import requests
from flask import Flask, request, jsonify
from PIL import Image
import boto3
import multiprocessing

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")
s3 = None
s3_bucket = 'data-bucket-sagemaker-image-search'
s3_key = 'training'
local_path = os.path.join(prefix, "processing", "input")

def initialize():
    global s3
    s3 = boto3.resource('s3')

def image_from_s3(inp):
    bucket, key = inp
    bucket = s3.Bucket(bucket)
    image = bucket.Object(key)
    img_data = image.get().get('Body').read()
    return Image.open(io.BytesIO(img_data))

class SearchService(object):
    filenames = None
    tree = None

    @classmethod
    def get_model(cls):
        if cls.tree == None:
            cls.tree, cls.filenames = [], []
            
            for pkl_file in glob.glob(os.path.join(model_path, 'metadata', '*.pkl')):
                with open(pkl_file, "rb") as f:
                    a, b = pickle.load(f)
                    cls.filenames += [a]
                    cls.tree += [b]
                
        return cls.tree

    @classmethod
    def predict(cls, embedding):
        SearchService.get_model()
        
        heap = []
        for i in range(len(cls.tree)):
            tree = cls.tree[i]
            dist, ind = tree.query(embedding, k=5)
            dist, ind = dist[0], ind[0]
            
            for j in range(len(dist)):
                if len(heap) < 5:
                    heapq.heappush(heap, (-dist[j], cls.filenames[i][ind[j]]))
                    
                elif dist[j] < -heap[0][0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, (-dist[j], cls.filenames[i][ind[j]]))
        
        return [y for x, y in heap]

app = flask.Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    health = SearchService.get_model() is not None

    status = 200 if health else 404
    return json.dumps({'status':status})

@app.route("/invocations", methods=["POST"])
def search():
    img_size = (128, 128)

    img = tf.keras.preprocessing.image.img_to_array(Image.open(BytesIO(base64.b64decode(request.form['b64']))).resize(img_size))

    payload = {
        "instances": [img.tolist()]
    }

    r = requests.post('http://localhost:8501/v1/models/image_search_model:predict', json=payload)
    embed = json.loads(r.content.decode('utf-8'))

    img_paths = SearchService.predict(embed['predictions'])
    
    if 'RUNTIME_ENV' not in os.environ or os.environ['RUNTIME_ENV'] == 'local':
        plt.figure(figsize=(10, 10))

        i = 0
        for file in img_paths:
            ax = plt.subplot(1, len(img_paths), i+1)
            img = tf.keras.preprocessing.image.load_img(os.path.join(local_path, file))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            plt.imshow(img_array / 255)
            plt.axis('off')
            i += 1
    else:
        img_inp = [(s3_bucket, os.path.join(s3_key, file)) for file in img_paths]

        pool = multiprocessing.Pool(multiprocessing.cpu_count(), initialize)
        images = pool.map(image_from_s3, img_inp)
        pool.close()
        pool.join()

        plt.figure(figsize=(10, 10))

        i = 0
        for img in images:
            ax = plt.subplot(1, len(img_paths), i+1)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            plt.imshow(img_array / 255)
            plt.axis('off')
            i += 1

    io_bytes = BytesIO()
    plt.savefig(io_bytes, format='jpg')
    io_bytes.seek(0)
    b64_data = base64.b64encode(io_bytes.read())

    return json.dumps({'status':'success', 'response':b64_data.decode('utf-8')})

