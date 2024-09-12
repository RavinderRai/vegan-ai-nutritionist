variable "region" {
  description = "The AWS region to deploy resources in"
}

variable "opensearch_domain_name" {
  description = "The name of the OpenSearch domain"
}

variable "instance_type" {
  description = "The type of instance for OpenSearch"
  default = "t3.small.search"  # Smaller, cheaper instance
}
