#!/bin/sh

dkr_image_name=$1

docker build -t ${dkr_image_name} .

export test_dir="${PWD}/local_test/test_dir"

cp -R ${PWD}/../prepare_data/local_test/test_dir local_test/
mv ${test_dir}/processing/output/* ${test_dir}/input/data/

cp ${PWD}/local_test/hyperparameters.json ${test_dir}/input/config/
cp ${PWD}/*.py ${test_dir}/code/

docker run --env RUNTIME_ENV="local" -v ${test_dir}:/opt/ml ${dkr_image_name} python3 train.py
# docker run -v ${test_dir}:/opt/ml -p 8080:8080 ${dkr_image_name} serve