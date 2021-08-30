#!/bin/sh

dkr_image_name=$1

docker build -t ${dkr_image_name} .

export test_dir="${PWD}/local_test/test_dir"

cp -R ${PWD}/../train/local_test/test_dir local_test/
cp ${PWD}/code/* ${test_dir}/code/

kubectl apply -f deployment.yaml 

# docker run --name image-search-deploy --env RUNTIME_ENV="local" -v ${test_dir}:/opt/ml -p 8080:8080 ${dkr_image_name} serve