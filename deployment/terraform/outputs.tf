# Outputs for SafePath Filter Terraform configuration

# Cluster information
output "cluster_id" {
  description = "EKS cluster ID"
  value       = module.eks.cluster_id
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = module.eks.cluster_arn
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "cluster_version" {
  description = "EKS cluster version"
  value       = module.eks.cluster_version
}

output "cluster_security_group_id" {
  description = "EKS cluster security group ID"
  value       = module.eks.cluster_security_group_id
}

output "cluster_oidc_issuer_url" {
  description = "The URL on the EKS cluster OIDC Issuer"
  value       = module.eks.cluster_oidc_issuer_url
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

# VPC information
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "VPC CIDR block"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "Public subnet IDs"
  value       = module.vpc.public_subnets
}

output "nat_gateway_ips" {
  description = "NAT gateway IPs"
  value       = module.vpc.nat_public_ips
}

# Database information
output "db_instance_endpoint" {
  description = "RDS instance endpoint"
  value       = var.enable_rds ? aws_db_instance.safepath[0].endpoint : null
  sensitive   = true
}

output "db_instance_id" {
  description = "RDS instance ID"
  value       = var.enable_rds ? aws_db_instance.safepath[0].id : null
}

output "db_instance_arn" {
  description = "RDS instance ARN"
  value       = var.enable_rds ? aws_db_instance.safepath[0].arn : null
}

output "db_subnet_group_id" {
  description = "Database subnet group ID"
  value       = aws_db_subnet_group.safepath.id
}

output "db_security_group_id" {
  description = "Database security group ID"
  value       = aws_security_group.safepath_db.id
}

# Cache information
output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = var.enable_redis ? aws_elasticache_replication_group.safepath[0].primary_endpoint_address : null
  sensitive   = true
}

output "redis_port" {
  description = "ElastiCache Redis port"
  value       = var.enable_redis ? aws_elasticache_replication_group.safepath[0].port : null
}

output "redis_security_group_id" {
  description = "Redis security group ID"
  value       = aws_security_group.safepath_cache.id
}

# Storage information
output "s3_bucket_id" {
  description = "S3 bucket ID for SafePath data"
  value       = aws_s3_bucket.safepath_data.id
}

output "s3_bucket_arn" {
  description = "S3 bucket ARN for SafePath data"
  value       = aws_s3_bucket.safepath_data.arn
}

output "s3_bucket_domain_name" {
  description = "S3 bucket domain name"
  value       = aws_s3_bucket.safepath_data.bucket_domain_name
}

# IAM information
output "app_role_arn" {
  description = "IAM role ARN for SafePath application"
  value       = aws_iam_role.safepath_app.arn
}

output "app_role_name" {
  description = "IAM role name for SafePath application"
  value       = aws_iam_role.safepath_app.name
}

# CloudWatch information
output "cloudwatch_log_group_name" {
  description = "CloudWatch log group name for SafePath"
  value       = aws_cloudwatch_log_group.safepath.name
}

output "cloudwatch_log_group_arn" {
  description = "CloudWatch log group ARN for SafePath"
  value       = aws_cloudwatch_log_group.safepath.arn
}

# Security information
output "db_password" {
  description = "Database password"
  value       = random_password.db_password.result
  sensitive   = true
}

output "redis_auth_token" {
  description = "Redis authentication token"
  value       = random_password.redis_auth.result
  sensitive   = true
}

# kubectl configuration
output "kubectl_config" {
  description = "kubectl config for accessing the cluster"
  value = {
    cluster_name = module.eks.cluster_id
    endpoint     = module.eks.cluster_endpoint
    region       = var.aws_region
    ca_data      = module.eks.cluster_certificate_authority_data
  }
  sensitive = true
}

# Helm deployment information
output "helm_values" {
  description = "Helm values for SafePath deployment"
  value = {
    image = {
      repository = "terragonlabs/cot-safepath-filter"
      tag        = "latest"
    }
    
    replicaCount = var.app_replica_count
    
    resources = {
      requests = {
        cpu    = var.app_cpu_request
        memory = var.app_memory_request
      }
      limits = {
        cpu    = var.app_cpu_limit
        memory = var.app_memory_limit
      }
    }
    
    autoscaling = {
      enabled     = true
      minReplicas = var.app_min_replicas
      maxReplicas = var.app_max_replicas
    }
    
    ingress = {
      enabled = true
      hosts = [
        {
          host = var.domain_name
          paths = [
            {
              path     = "/"
              pathType = "Prefix"
            }
          ]
        }
      ]
      tls = var.enable_ssl ? [
        {
          secretName = "${replace(var.domain_name, ".", "-")}-tls"
          hosts      = [var.domain_name]
        }
      ] : []
    }
    
    postgresql = {
      enabled = false  # Using external RDS
      external = var.enable_rds ? {
        host     = aws_db_instance.safepath[0].endpoint
        port     = aws_db_instance.safepath[0].port
        database = aws_db_instance.safepath[0].db_name
        username = aws_db_instance.safepath[0].username
      } : null
    }
    
    redis = {
      enabled = false  # Using external ElastiCache
      external = var.enable_redis ? {
        host = aws_elasticache_replication_group.safepath[0].primary_endpoint_address
        port = aws_elasticache_replication_group.safepath[0].port
      } : null
    }
    
    serviceAccount = {
      annotations = {
        "eks.amazonaws.com/role-arn" = aws_iam_role.safepath_app.arn
      }
    }
    
    monitoring = {
      enabled = var.enable_monitoring
    }
  }
}

# Environment-specific outputs
output "environment_info" {
  description = "Environment-specific information"
  value = {
    environment        = var.environment
    region            = var.aws_region
    cluster_name      = var.cluster_name
    domain_name       = var.domain_name
    monitoring_enabled = var.enable_monitoring
    ssl_enabled       = var.enable_ssl
    backup_enabled    = var.enable_backup
  }
}

# Connection strings (for application configuration)
output "connection_strings" {
  description = "Connection strings for external services"
  value = {
    database = var.enable_rds ? "postgresql://${aws_db_instance.safepath[0].username}:${random_password.db_password.result}@${aws_db_instance.safepath[0].endpoint}/${aws_db_instance.safepath[0].db_name}" : null
    redis    = var.enable_redis ? "redis://:${random_password.redis_auth.result}@${aws_elasticache_replication_group.safepath[0].primary_endpoint_address}:${aws_elasticache_replication_group.safepath[0].port}" : null
  }
  sensitive = true
}

# Resource ARNs for cross-referencing
output "resource_arns" {
  description = "ARNs of created resources"
  value = {
    cluster              = module.eks.cluster_arn
    vpc                 = module.vpc.vpc_arn
    database            = var.enable_rds ? aws_db_instance.safepath[0].arn : null
    s3_bucket           = aws_s3_bucket.safepath_data.arn
    app_role            = aws_iam_role.safepath_app.arn
    cloudwatch_log_group = aws_cloudwatch_log_group.safepath.arn
  }
}