locals {
  region = "us-east-1"
  tags = {
    Owner       = "amondal"
    Environment = "dev"
  }
}

provider "aws" {
  region = local.region
}

resource "aws_s3_bucket" "data_bucket" {
  bucket = "data-bucket-sagemaker-image-search"
  acl    = "private"

  tags = local.tags
}

resource "aws_s3_bucket" "model_bucket" {
  bucket = "model-bucket-sagemaker-image-search"
  acl    = "private"

  tags = local.tags
}

resource "aws_iam_role" "notebook_iam_role" {
  name = "sagemaker-notebook-role-image-search"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "",
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
}

resource "aws_iam_policy_attachment" "sm_full_access_attach" {
  name = "sagemaker-full-access-attachment-image-search"
  roles = [aws_iam_role.notebook_iam_role.name]
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_sagemaker_notebook_instance_lifecycle_configuration" "notebook_config" {
  name = "sagemaker-lifecycle-config-image-search"
  on_create = filebase64("../terraform-scripts/on-create.sh")
  on_start = filebase64("../terraform-scripts/on-start.sh")
}

resource "aws_sagemaker_code_repository" "git_repo" {
  code_repository_name = "image-search"
  
  git_config {
    repository_url = "https://github.com/funktor/image-search.git"
  }
}

resource "aws_sagemaker_notebook_instance" "notebook_instance" {
  name = "image-search-sagemaker-nb-instance"
  role_arn = aws_iam_role.notebook_iam_role.arn
  instance_type = "ml.t2.medium"
  lifecycle_config_name = aws_sagemaker_notebook_instance_lifecycle_configuration.notebook_config.name
  default_code_repository = aws_sagemaker_code_repository.git_repo.code_repository_name
}