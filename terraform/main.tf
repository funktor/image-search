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
  force_destroy = "true"

  tags = local.tags
}

resource "aws_s3_bucket" "model_bucket" {
  bucket = "model-bucket-sagemaker-image-search"
  acl    = "private"
  force_destroy = "true"

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

resource "aws_ecr_repository" "image-search-deploy_repository" {
  name                 = "sagemaker-image-search-deploy-repo"
  image_tag_mutability = "MUTABLE"
}

resource "aws_api_gateway_rest_api" "rest_api" {
  name = "image-search-api"
}

resource "aws_api_gateway_method" "predict" {
  rest_api_id      = aws_api_gateway_rest_api.rest_api.id
  resource_id      = aws_api_gateway_rest_api.rest_api.root_resource_id
  http_method      = "POST"
  authorization    = "NONE"
}

resource "aws_api_gateway_method_response" "predict_200" {
  rest_api_id = aws_api_gateway_rest_api.rest_api.id
  resource_id = aws_api_gateway_rest_api.rest_api.root_resource_id
  http_method = aws_api_gateway_method.predict.http_method
  status_code = "200"
}

data "aws_iam_policy_document" "apigw_sm_invoke_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["apigateway.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "apigw_sm_invoke" {
  name               = "image-search-sm-invoke"
  assume_role_policy = data.aws_iam_policy_document.apigw_sm_invoke_assume_role.json
}


data "aws_iam_policy_document" "apigw_sm_invoke_access" {
  statement {
    actions   = ["sagemaker:InvokeEndpoint"]
    resources = ["arn:aws:sagemaker:us-east-1:706073161968:endpoint/image-search-engine"]
  }
}

resource "aws_iam_role_policy" "apigw_sagemaker_invoke" {
  role   = aws_iam_role.apigw_sm_invoke.name
  policy = data.aws_iam_policy_document.apigw_sm_invoke_access.json
}

resource "aws_api_gateway_integration" "predict" {
  depends_on = [aws_iam_role_policy.apigw_sagemaker_invoke]

  rest_api_id = aws_api_gateway_rest_api.rest_api.id
  resource_id = aws_api_gateway_rest_api.rest_api.root_resource_id

  type                    = "AWS"
  http_method             = aws_api_gateway_method.predict.http_method
  integration_http_method = "POST"

  credentials = aws_iam_role.apigw_sm_invoke.arn
  uri         = "arn:aws:apigateway:${local.region}:runtime.sagemaker:path//endpoints/image-search-engine/invocations"
}

resource "aws_api_gateway_integration_response" "predict" {
  rest_api_id = aws_api_gateway_rest_api.rest_api.id
  resource_id = aws_api_gateway_rest_api.rest_api.root_resource_id
  http_method = aws_api_gateway_integration.predict.http_method
  status_code = aws_api_gateway_method_response.predict_200.status_code

  response_templates = {
    "application/json" = ""
  }
}

resource "aws_api_gateway_deployment" "predict" {
  depends_on = [aws_api_gateway_integration_response.predict]

  rest_api_id = aws_api_gateway_rest_api.rest_api.id
  stage_name  = "predict"
  description = "Image Search API"
}

output "invoke_url" {
  value = aws_api_gateway_deployment.predict.invoke_url
}