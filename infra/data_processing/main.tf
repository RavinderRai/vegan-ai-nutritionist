provider "aws" {
  region = var.region
}

resource "aws_opensearch_domain" "vector_database" {
  domain_name    = var.opensearch_domain_name
  engine_version = "OpenSearch_2.5"

  cluster_config {
    instance_type  = var.instance_type
    instance_count = 1  # Only 1 instance for lower cost
  }

  ebs_options {
    ebs_enabled = true         # Enable EBS for storage
    volume_size = 10           # 10 GB EBS volume, can adjust based on your needs
    volume_type = "gp2"        # General Purpose SSD (default)
  }
}

output "opensearch_endpoint" {
  value = aws_opensearch_domain.vector_database.endpoint
}
