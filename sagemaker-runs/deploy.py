import boto3, os, sagemaker, re, numpy as np
from sagemaker import get_execution_role
import sagemaker as sage
from time import gmtime, strftime

sagemaker_role = 'sagemaker-role-image-search'
s3_data_bucket = 'data-bucket-sagemaker-image-search'
s3_model_bucket = 'model-bucket-sagemaker-image-search'
s3_data_key = 'tfrecords'
sagemaker_ecr = 'sagemaker-image-search-deploy-repo:latest'
endpoint_name = 'image-search-engine'
model_path = "s3://{}/output/image-search-2021-08-30-08-14-18-650/output/model.tar.gz".format(s3_model_bucket)

role = boto3.client('iam').get_role(RoleName=sagemaker_role)['Role']['Arn']
sess = sage.Session()
account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
image = '{}.dkr.ecr.{}.amazonaws.com/{}'.format(account, region, sagemaker_ecr)

model = sage.model.Model(image_uri=image, 
                         model_data=model_path,
                         role=role, 
                         sagemaker_session=sess)

model.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge", endpoint_name=endpoint_name)

