#!/bin/sh

dkr_image_name=$1

docker build -t ${dkr_image_name} .

export test_dir="${PWD}/local_test/test_dir"

cp -R ${PWD}/../prepare_data/local_test/test_dir local_test/
mv ${test_dir}/processing/output/* ${test_dir}/input/data/

cp ${PWD}/local_test/hyperparameters.json ${test_dir}/input/config/
cp ${PWD}/code/*.py ${test_dir}/code/

docker run --env SM_MODEL_DIR="/opt/ml/model" --env SM_TRAINING_ENV='{"hyperparameters":{"batch_size":64,"epochs":2,"img_height":128,"img_width":128,"lr":0.001,"model_dir":"/opt/ml/model","run_id":1}, "channel_input_dirs":{"0":"/opt/ml/input/data/0","1":"/opt/ml/input/data/1","2":"/opt/ml/input/data/2","3":"/opt/ml/input/data/3","complete":"/opt/ml/input/data/complete"},"input_data_config":{"0":{"TrainingInputMode":"File"},"1":{"TrainingInputMode":"File"},"2":{"TrainingInputMode":"File"},"3":{"TrainingInputMode":"File"},"complete":{"TrainingInputMode":"File"}}}' --env SM_FRAMEWORK_PARAMS='{"sagemaker_mpi_enabled":false}' -v ${test_dir}:/opt/ml ${dkr_image_name} python3 train.py --model_dir="/opt/ml/model" --epochs=2

# docker run -v ${test_dir}:/opt/ml -p 8080:8080 ${dkr_image_name} serve