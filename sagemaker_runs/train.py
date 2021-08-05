import boto3, os, sagemaker, re, numpy as np
from sagemaker import get_execution_role
import sagemaker as sage
from time import gmtime, strftime

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

hyperparameters={'img_width':128, 'img_height':128, 'batch_size':32, 'epochs':2, 'lr':0.001, 'run_id':1}

model = sage.estimator.Estimator(image, 
                                 role, 
                                 instance_count=1, 
                                 instance_type='ml.c4.2xlarge', 
                                 output_path="s3://{}/output".format(s3_model_bucket), 
                                 base_job_name="image-search", 
                                 hyperparameters=hyperparameters,
                                 input_mode="Pipe",
                                 sagemaker_session=sess)

model.fit({'complete':'s3://{}/{}/complete.tfrecords'.format(s3_data_bucket, s3_data_key)})

print("Model S3 location", model.model_data)