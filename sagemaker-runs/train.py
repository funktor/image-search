import boto3, os, sagemaker, re, numpy as np
from sagemaker import get_execution_role
import sagemaker as sage
from time import gmtime, strftime
from sagemaker.tensorflow.estimator import TensorFlow
from sagemaker.session import s3_input

sagemaker_role = 'sagemaker-role-image-search'
s3_data_bucket = 'data-bucket-sagemaker-image-search'
s3_model_bucket = 'model-bucket-sagemaker-image-search'
s3_data_key = 'tfrecords'
sagemaker_ecr = 'sagemaker-image-search-repo:latest'

role = boto3.client('iam').get_role(RoleName=sagemaker_role)['Role']['Arn']
sess = sage.Session()
account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
image = '{}.dkr.ecr.{}.amazonaws.com/{}'.format(account, region, sagemaker_ecr)

hyperparameters={'img_width':128, 'img_height':128, 'batch_size':256, 'epochs':10, 'lr':0.001, 'run_id':1, 'is_parallel':"true"}

train_instance_type='ml.g4dn.12xlarge'
train_instance_count = 2
gpus_per_host = 4

distribution = {
    "mpi": {
        "enabled": True,
        "processes_per_host": gpus_per_host
    }
}

shuffle_config = sagemaker.session.ShuffleConfig(234)
train_s3_uri_prefix = 's3://{}/{}'.format(s3_data_bucket, s3_data_key)

remote_inputs = {}

for idx in range(gpus_per_host):
    train_s3_uri = f'{train_s3_uri_prefix}/{idx}/'
    train_s3_input = s3_input(train_s3_uri, shuffle_config=shuffle_config, distribution='ShardedByS3Key')
    remote_inputs[f'{idx}'] = train_s3_input

model = TensorFlow(image_uri=image, 
                   role=role, 
                   entry_point='train.py',
                   instance_count=train_instance_count, 
                   instance_type=train_instance_type, 
                   output_path="s3://{}/output".format(s3_model_bucket), 
                   base_job_name="image-search", 
                   hyperparameters=hyperparameters, 
                   distribution=distribution,
                   input_mode="Pipe")

model.fit(remote_inputs)

print("Model S3 location", model.model_data)