provider "aws" {
    region = var.region
}

resource "aws_s3_bucket" "train-test-bucket" {
    bucket = var.training_data_bucket_name

    tags = {
        Name = "SageMaker Training Data Bucket"
        Environment = "Development"
    }
}

resource "aws_s3_bucket" "model-bucket" {
    bucket = var.model_artifact_bucket_name

    tags = {
        Name = "SageMaker Model Artifact Bucket"
        Environment = "Development"
    }
}

resource "aws_s3_bucket_policy" "sagemaker_bucket_policy" {
  bucket = aws_s3_bucket.train-test-bucket.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "s3:GetObject"
        Resource = "${aws_s3_bucket.train-test-bucket.arn}/*"
      },
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "s3:PutObject"
        Resource = "${aws_s3_bucket.train-test-bucket.arn}/*"
      }
    ]
  })
}


# resource "aws_sagemaker_training_job" "fine-tune" {
#   name = "fine-tuning-job"

#   role_arn = var.sagemaker_role_arn

#   input_data_config {
#     channel_name = "training"
#     data_source {
#       s3_data_source {
#         s3_data_type = "S3Prefix"
#         s3_uri = aws_s3_bucket.training_data_bucket.arn
#       }
#     }
#   }

#   output_data_config {
#     s3_output_path = aws_s3_bucket.output_data_bucket.arn
#   }

#   resource_config {
#     instance_type = "ml.p2.xlarge" 
#     instance_count = 1
#     volume_size = 20  # Size in GB

#     spot_price = "0.50"  # Set your maximum price for the spot instance
#   }

#   enable_network_isolation = false  # Set to true if you want to isolate the training job from the internet
# }
