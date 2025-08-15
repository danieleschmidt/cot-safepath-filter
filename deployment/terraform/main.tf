# Terraform configuration for SafePath Filter infrastructure
terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
  
  backend "s3" {
    bucket         = "terragon-terraform-state"
    key            = "safepath/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local values
locals {
  name         = var.cluster_name
  region       = var.aws_region
  environment  = var.environment
  
  vpc_cidr = "10.0.0.0/16"
  azs      = slice(data.aws_availability_zones.available.names, 0, 3)
  
  tags = {
    Name        = local.name
    Environment = local.environment
    Project     = "safepath-filter"
    Owner       = "terragon-labs"
    ManagedBy   = "terraform"
  }
}

# VPC Module
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = local.name
  cidr = local.vpc_cidr
  
  azs             = local.azs
  private_subnets = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 4, k)]
  public_subnets  = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 8, k + 48)]
  intra_subnets   = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 8, k + 52)]
  
  enable_nat_gateway   = true
  single_nat_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  # Kubernetes tags
  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }
  
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }
  
  tags = local.tags
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = local.name
  cluster_version = var.kubernetes_version
  
  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true
  
  # Cluster access entry
  enable_cluster_creator_admin_permissions = true
  
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }
  
  # EKS Managed Node Groups
  eks_managed_node_groups = {
    general = {
      min_size     = 2
      max_size     = 10
      desired_size = 3
      
      instance_types = ["t3.large"]
      capacity_type  = "ON_DEMAND"
      
      update_config = {
        max_unavailable_percentage = 33
      }
      
      labels = {
        role = "general"
      }
    }
    
    safepath = {
      min_size     = 3
      max_size     = 20
      desired_size = 5
      
      instance_types = ["c5.xlarge"]
      capacity_type  = "ON_DEMAND"
      
      update_config = {
        max_unavailable_percentage = 25
      }
      
      labels = {
        role = "safepath"
        workload = "ai-filtering"
      }
      
      taints = {
        dedicated = {
          key    = "safepath"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
    }
  }
  
  tags = local.tags
}

# RDS Instance for SafePath data
resource "aws_db_subnet_group" "safepath" {
  name       = "${local.name}-db-subnet-group"
  subnet_ids = module.vpc.private_subnets
  
  tags = merge(local.tags, {
    Name = "${local.name}-db-subnet-group"
  })
}

resource "aws_security_group" "safepath_db" {
  name_prefix = "${local.name}-db-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [local.vpc_cidr]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.tags, {
    Name = "${local.name}-db-security-group"
  })
}

resource "random_password" "db_password" {
  length  = 16
  special = true
}

resource "aws_db_instance" "safepath" {
  count = var.enable_rds ? 1 : 0
  
  identifier = "${local.name}-database"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.db_instance_class
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type          = "gp3"
  storage_encrypted     = true
  
  db_name  = "safepath"
  username = "safepath_admin"
  password = random_password.db_password.result
  
  vpc_security_group_ids = [aws_security_group.safepath_db.id]
  db_subnet_group_name   = aws_db_subnet_group.safepath.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = var.environment != "production"
  deletion_protection = var.environment == "production"
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  
  tags = merge(local.tags, {
    Name = "${local.name}-database"
  })
}

# ElastiCache Redis for SafePath caching
resource "aws_elasticache_subnet_group" "safepath" {
  name       = "${local.name}-cache-subnet-group"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "safepath_cache" {
  name_prefix = "${local.name}-cache-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [local.vpc_cidr]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.tags, {
    Name = "${local.name}-cache-security-group"
  })
}

resource "aws_elasticache_replication_group" "safepath" {
  count = var.enable_redis ? 1 : 0
  
  replication_group_id       = "${local.name}-redis"
  description                = "Redis cluster for SafePath Filter"
  
  node_type = var.redis_node_type
  port      = 6379
  
  num_cache_clusters = 3
  
  engine_version     = "7.0"
  parameter_group_name = "default.redis7"
  
  subnet_group_name  = aws_elasticache_subnet_group.safepath.name
  security_group_ids = [aws_security_group.safepath_cache.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = random_password.redis_auth.result
  
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_slow.name
    destination_type = "cloudwatch-logs"
    log_format       = "text"
    log_type         = "slow-log"
  }
  
  tags = local.tags
}

resource "random_password" "redis_auth" {
  length  = 32
  special = false
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "safepath" {
  name              = "/aws/eks/${local.name}/safepath"
  retention_in_days = 30
  
  tags = local.tags
}

resource "aws_cloudwatch_log_group" "redis_slow" {
  name              = "/aws/elasticache/redis/${local.name}/slow-log"
  retention_in_days = 14
  
  tags = local.tags
}

# S3 Bucket for SafePath data and backups
resource "aws_s3_bucket" "safepath_data" {
  bucket = "${local.name}-safepath-data-${random_id.bucket_suffix.hex}"
  
  tags = merge(local.tags, {
    Name = "${local.name}-safepath-data"
  })
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_versioning" "safepath_data" {
  bucket = aws_s3_bucket.safepath_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "safepath_data" {
  bucket = aws_s3_bucket.safepath_data.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "safepath_data" {
  bucket = aws_s3_bucket.safepath_data.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# IAM Role for SafePath application
resource "aws_iam_role" "safepath_app" {
  name = "${local.name}-safepath-app-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Condition = {
          StringEquals = {
            "${replace(module.eks.oidc_provider_arn, "/^(.*provider/)/", "")}:sub" = "system:serviceaccount:safepath:safepath-service-account"
            "${replace(module.eks.oidc_provider_arn, "/^(.*provider/)/", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })
  
  tags = local.tags
}

resource "aws_iam_policy" "safepath_app" {
  name        = "${local.name}-safepath-app-policy"
  description = "IAM policy for SafePath application"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.safepath_data.arn,
          "${aws_s3_bucket.safepath_data.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = [
          aws_cloudwatch_log_group.safepath.arn,
          "${aws_cloudwatch_log_group.safepath.arn}:*"
        ]
      }
    ]
  })
  
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "safepath_app" {
  role       = aws_iam_role.safepath_app.name
  policy_arn = aws_iam_policy.safepath_app.arn
}