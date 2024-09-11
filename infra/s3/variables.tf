variable "bucket_name" {
  description = "S3 bucket to hold raw pdf meta data and text data"
  type        = string
}

variable "region" {
  description = "AWS region where resources will be created"
  default     = "us-east-1"
}
