#!/bin/sh

dkr_image_name=$1
images_folder=$2

docker build -t ${dkr_image_name} .

export test_dir="${PWD}/local_test/test_dir"

mkdir -p ${test_dir}
mkdir -p ${test_dir}/input
mkdir -p ${test_dir}/input/data
mkdir -p ${test_dir}/input/config
mkdir -p ${test_dir}/input/data/train
mkdir -p ${test_dir}/model
mkdir -p ${test_dir}/output
mkdir -p ${test_dir}/code
mkdir -p ${test_dir}/processing
mkdir -p ${test_dir}/processing/input
mkdir -p ${test_dir}/processing/output

cp -R ${images_folder}/* ${test_dir}/processing/input/
cp ${PWD}/code/* ${test_dir}/processing/

docker run -v ${test_dir}:/opt/ml ${dkr_image_name} "--input=/opt/ml/processing/input" "--output=/opt/ml/processing/output" "--number_instances=2" "--num_gpus_per_instance=4"