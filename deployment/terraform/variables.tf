# Variables for SafePath Filter Terraform configuration

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "safepath-filter"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "production"
  
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

# Database configuration
variable "enable_rds" {
  description = "Enable RDS PostgreSQL database"
  type        = bool
  default     = true
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "db_allocated_storage" {
  description = "Initial allocated storage for RDS"
  type        = number
  default     = 100
}

variable "db_max_allocated_storage" {
  description = "Maximum allocated storage for RDS"
  type        = number
  default     = 1000
}

# Cache configuration
variable "enable_redis" {
  description = "Enable ElastiCache Redis"
  type        = bool
  default     = true
}

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.medium"
}

variable "redis_num_cache_clusters" {
  description = "Number of cache clusters for Redis"
  type        = number
  default     = 3
}

# Application configuration
variable "app_replica_count" {
  description = "Number of application replicas"
  type        = number
  default     = 3
}

variable "app_min_replicas" {
  description = "Minimum number of replicas for autoscaling"
  type        = number
  default     = 3
}

variable "app_max_replicas" {
  description = "Maximum number of replicas for autoscaling"
  type        = number
  default     = 20
}

variable "app_cpu_request" {
  description = "CPU request for application pods"
  type        = string
  default     = "250m"
}

variable "app_memory_request" {
  description = "Memory request for application pods"
  type        = string
  default     = "512Mi"
}

variable "app_cpu_limit" {
  description = "CPU limit for application pods"
  type        = string
  default     = "1000m"
}

variable "app_memory_limit" {
  description = "Memory limit for application pods"
  type        = string
  default     = "2Gi"
}

# Monitoring configuration
variable "enable_monitoring" {
  description = "Enable monitoring stack (Prometheus/Grafana)"
  type        = bool
  default     = true
}

variable "enable_logging" {
  description = "Enable centralized logging"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

# Security configuration
variable "enable_pod_security_policy" {
  description = "Enable Pod Security Policy"
  type        = bool
  default     = true
}

variable "enable_network_policy" {
  description = "Enable Network Policies"
  type        = bool
  default     = true
}

variable "enable_service_mesh" {
  description = "Enable service mesh (Istio)"
  type        = bool
  default     = false
}

# Backup configuration
variable "enable_backup" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 7
}

# Domain configuration
variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "safepath.terragonlabs.com"
}

variable "enable_ssl" {
  description = "Enable SSL/TLS with cert-manager"
  type        = bool
  default     = true
}

variable "ssl_issuer" {
  description = "cert-manager issuer for SSL certificates"
  type        = string
  default     = "letsencrypt-prod"
}

# Development and testing
variable "enable_debug" {
  description = "Enable debug mode"
  type        = bool
  default     = false
}

variable "enable_testing" {
  description = "Enable testing resources"
  type        = bool
  default     = false
}

# Cost optimization
variable "enable_spot_instances" {
  description = "Enable spot instances for worker nodes"
  type        = bool
  default     = false
}

variable "spot_instance_types" {
  description = "Instance types for spot instances"
  type        = list(string)
  default     = ["c5.large", "c5.xlarge", "m5.large", "m5.xlarge"]
}

# Compliance and governance
variable "enable_compliance_scanning" {
  description = "Enable compliance scanning"
  type        = bool
  default     = true
}

variable "enable_vulnerability_scanning" {
  description = "Enable vulnerability scanning"
  type        = bool
  default     = true
}

# Data protection
variable "enable_encryption" {
  description = "Enable encryption at rest and in transit"
  type        = bool
  default     = true
}

variable "kms_key_id" {
  description = "KMS key ID for encryption (optional)"
  type        = string
  default     = ""
}

# Disaster recovery
variable "enable_cross_region_backup" {
  description = "Enable cross-region backup"
  type        = bool
  default     = false
}

variable "backup_region" {
  description = "Region for cross-region backup"
  type        = string
  default     = "us-east-1"
}

# Performance tuning
variable "enable_performance_insights" {
  description = "Enable RDS Performance Insights"
  type        = bool
  default     = true
}

variable "enable_enhanced_monitoring" {
  description = "Enable enhanced monitoring for RDS"
  type        = bool
  default     = true
}

# Tags
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}