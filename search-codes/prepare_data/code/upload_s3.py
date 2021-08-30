import tensorflow as tf
import numpy as np
import os, math, random
import os
import multiprocessing
import boto3

S3_BUCKET_NAME = 'data-bucket-sagemaker-image-search'
images_dir = "/Users/abhijitmondal/Documents/datasets/natural_images"
s3 = None

extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def initialize():
    global s3
    s3 = boto3.resource('s3')

def upload_to_s3(inp):
    file_path, key = inp
    print(key)
    s3.meta.client.upload_file(file_path, S3_BUCKET_NAME, key) 
    
def upload():
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(S3_BUCKET_NAME)
    keys = set([file.key for file in my_bucket.objects.all()])
    
    inp = []
    for root, directories, filenames in os.walk(images_dir):
        for d in directories:
            for filename in os.listdir(os.path.join(root, d)):
                if any(ext in filename for ext in extensions) and os.path.join("training", d, filename) not in keys:
                    inp.append((os.path.join(root, d, filename), os.path.join("training", d, filename)))
    
    pool = multiprocessing.Pool(10, initialize)
    pool.map(upload_to_s3, inp)
    pool.close()
    pool.join()
    
if __name__=="__main__":
    upload()