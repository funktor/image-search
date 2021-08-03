#!/bin/sh

dkr_image_name=$1

docker build -t ${dkr_image_name} .

export test_dir="${PWD}/local_test/test_dir"

cp -R ${PWD}/im_search_code/prepare_data/local_test/test_dir local_test/

cp ${PWD}/local_test/hyperparameters.json ${test_dir}/input/config/
cp ${PWD}/im_search_code/* ${test_dir}/code/

docker run --env environment="local" -v ${test_dir}:/opt/ml ${dkr_image_name} train
docker run -v ${test_dir}:/opt/ml -p 8080:8080 ${dkr_image_name} serve