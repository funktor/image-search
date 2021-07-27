#!/bin/bash

set -e

## IDLE AUTOSTOP STEPS
## ----------------------------------------------------------------

# Setting the timeout (in seconds) for how long the SageMaker notebook can run idly before being auto-stopped
IDLE_TIME=300

# Getting the autostop.py script from GitHub
echo "Fetching the autostop script..."
wget https://raw.githubusercontent.com/funktor/image-search/main/terraform-scripts/autostop.py

# Using crontab to autostop the notebook when idle time is breached
echo "Starting the SageMaker autostop script in cron."
(crontab -l 2>/dev/null; echo "*/5 * * * * /usr/bin/python $PWD/autostop.py --time $IDLE_TIME --ignore-connections") | crontab -

NOTEBOOK_INSTANCE_NAME=$(jq '.ResourceName' \
                      /opt/ml/metadata/resource-metadata.json --raw-output)

echo "Fetching the CloudWatch agent configuration file."
wget https://raw.githubusercontent.com/funktor/image-search/main/terraform-scripts/amazon-cloudwatch-agent.json

sed -i -- "s/MyNotebookInstance/$NOTEBOOK_INSTANCE_NAME/g" amazon-cloudwatch-agent.json

echo "Starting the CloudWatch agent on the Notebook Instance."
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a \
    fetch-config -m ec2 -c file://$(pwd)/amazon-cloudwatch-agent.json -s

rm amazon-cloudwatch-agent.json

YOUR_USER_NAME="funktor"
YOUR_EMAIL_ADDRESS="abhi2iitk@gmail.com"

## CUSTOM CONDA KERNEL USAGE STEPS
## ----------------------------------------------------------------

# Setting the proper user credentials
sudo -u ec2-user -i <<'EOF'
unset SUDO_UID

git config --global user.name "$YOUR_USER_NAME"
git config --global user.email "$YOUR_EMAIL_ADDRESS"

# Setting the source for the custom conda kernel
WORKING_DIR=/home/ec2-user/SageMaker/custom-miniconda
source "$WORKING_DIR/miniconda/bin/activate"
# Loading all the custom kernels
for env in $WORKING_DIR/miniconda/envs/*; do
    BASENAME=$(basename "$env")
    source activate "$BASENAME"
    python -m ipykernel install --user --name "$BASENAME" --display-name "Custom ($BASENAME)"
done

