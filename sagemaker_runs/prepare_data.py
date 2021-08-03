import boto3, os, sagemaker, re, numpy as np
from sagemaker import get_execution_role
import sagemaker as sage
from time import gmtime, strftime
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput

sagemaker_role = 'sagemaker-role-image-search'
s3_data_bucket = 'data-bucket-sagemaker-image-search'
s3_model_bucket = 'model-bucket-sagemaker-image-search'
s3_data_key = 'training'
s3_tfrecords_key = 'tfrecords'
sagemaker_ecr = 'sagemaker-tfrecords-data-repo:latest'

role = boto3.client('iam').get_role(RoleName=sagemaker_role)['Role']['Arn']
sess = sage.Session()
account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
image = '{}.dkr.ecr.{}.amazonaws.com/{}'.format(account, region, sagemaker_ecr)

data_processor = Processor(role=role, 
                           image_uri=image, 
                           instance_count=1, 
                           instance_type='ml.m5.xlarge',
                           volume_size_in_gb=30, 
                           max_runtime_in_seconds=1200,
                           base_job_name='image-search-data-tfrecords')

input_folder = '/opt/ml/processing/input'
output_folder = '/opt/ml/processing/output'
s3_input = 's3://{}/{}'.format(s3_data_bucket, s3_data_key)

data_processor.run(
    arguments= [
        f'--input={input_folder}',
        f'--output={output_folder}'
    ],
    inputs = [
        ProcessingInput(
            input_name='input',
            source=s3_input,
            destination=input_folder
        )
    ],
    outputs= [
        ProcessingOutput(
            output_name='tfrecords',
            source=output_folder,
            destination=f's3://{s3_data_bucket}/{s3_tfrecords_key}'
        )
    ]
)