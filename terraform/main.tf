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

resource "aws_iam_role" "sagemaker_iam_role" {
  name = "sagemaker-role-image-search"

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
  roles = [aws_iam_role.sagemaker_iam_role.name]
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_ecr_repository" "image-search_repository" {
  name                 = "sagemaker-image-search-repo"
  image_tag_mutability = "MUTABLE"
}

resource "aws_ecr_repository" "tfrecords_repository" {
  name                 = "sagemaker-tfrecords-data-repo"
  image_tag_mutability = "MUTABLE"
}