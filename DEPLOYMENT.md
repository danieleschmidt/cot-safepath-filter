# 🚀 CoT SafePath Filter - Production Deployment Guide

Complete guide for deploying CoT SafePath Filter in production environments with global-first architecture.

## 📋 Prerequisites

### System Requirements
- **CPU**: 4+ cores (8 recommended for high traffic)
- **Memory**: 8GB minimum (16GB+ recommended)
- **Storage**: 100GB+ SSD for database and logs
- **Network**: 1Gbps bandwidth for high-throughput scenarios

### Software Dependencies
- Docker 20.10+
- Docker Compose 2.0+
- PostgreSQL 15+ (or managed database)
- Redis 7+ (or managed cache)
- Load balancer (Nginx/HAProxy/cloud LB)

## 🌍 Global Deployment Architecture

### Multi-Region Setup
```
┌─────────────────────────────────────────────────────────────┐
│                    Global Load Balancer                     │
│                   (CloudFlare/AWS ALB)                     │
└─────────────────┬───────────────┬───────────────┬───────────┘
                 │               │               │
         ┌───────v────┐  ┌───────v────┐  ┌───────v────┐
         │   US-East  │  │   EU-West  │  │  AP-South  │
         │   Region   │  │   Region   │  │   Region   │
         └────────────┘  └────────────┘  └────────────┘
```

### Regional Components
Each region contains:
- 3+ SafePath Filter instances (High Availability)
- PostgreSQL Primary + Read Replicas
- Redis Cluster (3 masters + 3 replicas)
- Prometheus + Grafana monitoring
- Log aggregation (ELK/Fluentd)

## 🔧 Quick Start - Single Region

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/terragonlabs/cot-safepath-filter
cd cot-safepath-filter

# Set environment variables
cp .env.example .env
# Edit .env with your configuration
```

### 2. Production Environment Variables
```bash
# Core Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
PORT=8080

# Database
DATABASE_URL=postgresql://safepath:secure_password@postgres:5432/safepath
POSTGRES_PASSWORD=your_secure_password_here

# Cache
REDIS_URL=redis://redis:6379/0

# Security
JWT_SECRET_KEY=your-256-bit-secret-key
ENCRYPTION_KEY=your-base64-encryption-key

# Monitoring
PROMETHEUS_ENABLED=true
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project

# Performance
MAX_CONCURRENT_REQUESTS=100
THREAD_POOL_SIZE=16
CACHE_SIZE_MB=512
```

### 3. Deploy Services
```bash
# Build and start production services
docker-compose --profile production up -d

# Verify all services are healthy
docker-compose ps
docker-compose logs safepath-filter
```

### 4. Initialize Database
```bash
# Run database migrations
docker-compose exec safepath-filter safepath-migrate

# Create initial admin user
docker-compose exec safepath-filter safepath-create-user \
  --username admin \
  --email admin@yourcompany.com \
  --role admin
```

### 5. Configure Monitoring
```bash
# Access Grafana dashboard
open http://localhost:3000
# Login: admin / (password from GRAFANA_PASSWORD)

# Access Prometheus metrics
open http://localhost:9091

# View application logs
docker-compose logs -f safepath-filter
```

## 🏗️ Production Infrastructure

### Docker Compose Production
```bash
# Start production stack
docker-compose --file docker-compose.yml --profile production up -d

# Scale application instances
docker-compose up -d --scale safepath-filter=3

# Rolling updates
docker-compose up -d --no-deps safepath-filter
```

### Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/safepath-filter.yaml
kubectl apply -f k8s/monitoring.yaml

# Verify deployment
kubectl get pods -n safepath
kubectl get services -n safepath
kubectl describe deployment safepath-filter -n safepath
```

### Cloud Provider Specific

#### AWS ECS/EKS
```bash
# ECS Task Definition
aws ecs register-task-definition --cli-input-json file://aws/ecs-task-definition.json

# EKS with Helm
helm install safepath-filter ./helm/safepath-filter \
  --namespace safepath \
  --set image.tag=v1.0.0 \
  --set replicaCount=3
```

#### Google Cloud Run/GKE
```bash
# Cloud Run deployment
gcloud run deploy safepath-filter \
  --image gcr.io/project/cot-safepath-filter:latest \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10

# GKE with terraform
terraform init
terraform plan -var="project_id=your-project"
terraform apply
```

#### Azure Container Instances/AKS
```bash
# ACI deployment
az container create \
  --resource-group safepath-rg \
  --name safepath-filter \
  --image cot-safepath-filter:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8080

# AKS deployment
az aks get-credentials --resource-group safepath-rg --name safepath-cluster
kubectl apply -f azure/
```

## 🔒 Security Configuration

### SSL/TLS Setup
```nginx
# Nginx configuration
server {
    listen 443 ssl http2;
    server_name api.safepath.yourcompany.com;
    
    ssl_certificate /etc/ssl/certs/safepath.crt;
    ssl_certificate_key /etc/ssl/private/safepath.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    
    location / {
        proxy_pass http://safepath-filter:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Database Security
```sql
-- Create read-only user for monitoring
CREATE USER safepath_monitor WITH PASSWORD 'monitor_password';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO safepath_monitor;

-- Set up row-level security
ALTER TABLE filter_operations ENABLE ROW LEVEL SECURITY;
CREATE POLICY user_filter_operations ON filter_operations 
  FOR ALL TO safepath_api USING (user_id = current_setting('app.current_user_id'));
```

### API Security
```yaml
# Rate limiting configuration
rate_limits:
  anonymous: 100/hour
  authenticated: 1000/hour
  premium: 10000/hour

# API key management
api_keys:
  length: 32
  prefix: "sk-"
  expiration_days: 365
  rate_limit_overrides: true
```

## 📊 Monitoring & Observability

### Health Checks
```bash
# Application health
curl -f http://localhost:8080/health

# Database health
curl -f http://localhost:8080/health/database

# Cache health
curl -f http://localhost:8080/health/cache

# Comprehensive health
curl -f http://localhost:8080/health/full
```

### Metrics Collection
```yaml
# Prometheus scrape configuration
scrape_configs:
  - job_name: 'safepath-filter'
    static_configs:
      - targets: ['safepath-filter:8080']
    metrics_path: /metrics
    scrape_interval: 15s
    scrape_timeout: 10s
```

### Log Management
```yaml
# Fluentd configuration
<source>
  @type docker
  tag docker.safepath.*
  format json
</source>

<match docker.safepath.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name safepath-logs
  type_name _doc
</match>
```

## 🔄 CI/CD Pipeline

### GitHub Actions
```yaml
name: Deploy to Production
on:
  push:
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v3
      
      - name: Build Docker Image
        run: |
          docker build -t cot-safepath-filter:${{ github.ref_name }} .
          docker tag cot-safepath-filter:${{ github.ref_name }} cot-safepath-filter:latest
      
      - name: Run Quality Gates
        run: python run_quality_gates.py
      
      - name: Deploy to Production
        run: |
          docker-compose --profile production up -d
          ./scripts/health-check.sh
```

### Blue-Green Deployment
```bash
#!/bin/bash
# Blue-Green deployment script

# Deploy to green environment
docker-compose -f docker-compose.green.yml up -d

# Wait for health check
./scripts/wait-for-health.sh green

# Switch traffic to green
./scripts/switch-traffic.sh blue green

# Health check new environment
./scripts/health-check.sh

# Cleanup old blue environment
docker-compose -f docker-compose.blue.yml down
```

## 🌐 Global Configuration

### Multi-Language Support
```yaml
# i18n configuration
languages:
  - code: en
    name: English
    default: true
  - code: es
    name: Español
  - code: fr
    name: Français
  - code: de
    name: Deutsch
  - code: ja
    name: 日本語
  - code: zh
    name: 中文

# Regional compliance
compliance:
  gdpr:
    enabled: true
    regions: [EU]
  ccpa:
    enabled: true
    regions: [US-CA]
  pdpa:
    enabled: true
    regions: [SG, MY, TH]
```

### Regional Scaling
```yaml
# Auto-scaling configuration
scaling:
  regions:
    us-east-1:
      min_instances: 3
      max_instances: 20
      target_cpu: 70%
    eu-west-1:
      min_instances: 2
      max_instances: 15
      target_cpu: 70%
    ap-southeast-1:
      min_instances: 2
      max_instances: 10
      target_cpu: 70%
```

## 🔧 Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
docker stats safepath-filter

# Analyze heap dump
docker exec safepath-filter python -m memory_profiler app.py

# Tune JVM/Python settings
export MEMORY_LIMIT=4g
export CACHE_SIZE_MB=1024
```

#### Database Connection Issues
```bash
# Check database connectivity
docker exec safepath-filter pg_isready -h postgres -U safepath

# Monitor connection pool
docker exec safepath-filter psql -h postgres -U safepath -c "
  SELECT state, count(*) 
  FROM pg_stat_activity 
  WHERE datname='safepath' 
  GROUP BY state;"
```

#### Performance Degradation
```bash
# Check application metrics
curl http://localhost:8080/metrics | grep safepath_

# Profile slow requests
docker exec safepath-filter python -m cProfile -o profile.stats app.py

# Analyze query performance
docker exec postgres psql -U safepath -c "
  SELECT query, mean_time, calls 
  FROM pg_stat_statements 
  ORDER BY mean_time DESC 
  LIMIT 10;"
```

### Log Analysis
```bash
# Search for errors
docker-compose logs safepath-filter | grep ERROR

# Monitor response times
docker-compose logs safepath-filter | grep "processing_time_ms" | tail -100

# Check security events
docker-compose logs safepath-filter | grep "security_event"
```

## 🚀 Performance Optimization

### Database Optimization
```sql
-- Create indexes for common queries
CREATE INDEX CONCURRENTLY idx_filter_ops_created_at_user 
  ON filter_operations (created_at, user_id);

CREATE INDEX CONCURRENTLY idx_safety_detections_operation_detector 
  ON safety_detections (operation_id, detector_name);

-- Optimize PostgreSQL settings
ALTER SYSTEM SET shared_buffers = '512MB';
ALTER SYSTEM SET effective_cache_size = '2GB';
ALTER SYSTEM SET max_connections = '200';
SELECT pg_reload_conf();
```

### Cache Optimization
```redis
# Redis configuration
CONFIG SET maxmemory 1gb
CONFIG SET maxmemory-policy allkeys-lru
CONFIG SET save "900 1 300 10 60 10000"
```

### Application Optimization
```yaml
# Performance tuning
performance:
  max_concurrent_requests: 100
  thread_pool_size: 16
  request_timeout_ms: 5000
  cache:
    size_mb: 512
    ttl_seconds: 3600
  batch_processing:
    enabled: true
    batch_size: 10
    timeout_ms: 100
```

## 📞 Support & Maintenance

### Backup Strategy
```bash
# Database backup
docker exec postgres pg_dump -U safepath safepath > backup_$(date +%Y%m%d).sql

# Redis backup
docker exec redis redis-cli BGSAVE
docker cp redis:/data/dump.rdb ./backup_redis_$(date +%Y%m%d).rdb

# Configuration backup
tar -czf config_backup_$(date +%Y%m%d).tar.gz .env docker-compose.yml monitoring/
```

### Update Procedure
```bash
# 1. Backup current state
./scripts/backup.sh

# 2. Update application
git pull origin main
docker-compose pull

# 3. Run quality gates
python run_quality_gates.py

# 4. Deploy with rolling update
docker-compose up -d --no-deps safepath-filter

# 5. Verify deployment
./scripts/health-check.sh
```

### Emergency Procedures
```bash
# Scale down during incident
docker-compose up -d --scale safepath-filter=1

# Enable maintenance mode
curl -X POST http://localhost:8080/admin/maintenance \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Emergency rollback
git checkout v1.0.0
docker-compose up -d --no-deps safepath-filter
```

## 📧 Contact & Support

- **Technical Support**: support@terragonlabs.com
- **Security Issues**: security@terragonlabs.com  
- **Documentation**: https://docs.safepath.terragonlabs.com
- **Status Page**: https://status.safepath.terragonlabs.com

---

## 📄 License

CoT SafePath Filter is licensed under the MIT License. See [LICENSE](LICENSE) for details.

**Production Deployment Complete! 🎉**