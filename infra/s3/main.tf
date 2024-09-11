provider "aws" {
  region = var.region
}

resource "aws_s3_bucket" "raw-pdf-data" {
  bucket = var.bucket_name
}
