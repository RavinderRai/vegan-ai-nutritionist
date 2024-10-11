variable "training_data_bucket_name" {
  description = "S3 bucket to hold training and testing data for fine-tuning"
  type        = string
}

variable "model_artifact_bucket_name" {
  description = "S3 bucket to hold the model after training"
  type        = string
}

variable "region" {
  description = "AWS region where resources will be created"
  default     = "us-east-1"
}

variable "sagemaker_role_arn" {
  description = "The ARN of the existing SageMaker IAM role"
  type        = string
}
