"""
Production deployment configuration for CoT SafePath Filter.

This module provides production-ready configurations for deploying
the SafePath Filter system with optimal performance and reliability.
"""

from cot_safepath.performance_optimization import OptimizationConfig
from cot_safepath.models import SafetyLevel
from cot_safepath.enhanced_integrations import IntegrationConfig
from cot_safepath.realtime_monitoring import MonitoringConfig
import os

# Production Environment Configuration
PRODUCTION_CONFIG = {
    "environment": "production",
    "version": "3.0.0",
    "deployment_date": "2025-08-27",
    "service_name": "cot-safepath-filter",
    "namespace": "ai-safety"
}

# High-Performance Production Configuration
PRODUCTION_OPTIMIZATION_CONFIG = OptimizationConfig(
    # Caching Configuration
    enable_caching=True,
    cache_size_mb=500,  # 500MB cache for production workloads
    cache_ttl_seconds=3600,  # 1 hour cache TTL
    
    # Concurrency Configuration  
    max_concurrent_requests=1000,  # Support 1000 concurrent requests
    max_worker_threads=16,  # 16 worker threads for CPU-intensive tasks
    max_worker_processes=8,  # 8 processes for parallel processing
    
    # Connection Pooling
    enable_connection_pooling=True,
    pool_size=20,  # 20 connection pool size
    pool_timeout_seconds=30,
    
    # Performance Optimization
    enable_adaptive_tuning=True,
    performance_target_ms=50.0,  # Target 50ms response time
    auto_scaling_enabled=True,
    scale_up_threshold=0.75,  # Scale up at 75% utilization
    scale_down_threshold=0.25,  # Scale down at 25% utilization
)

# Production Integration Configuration
PRODUCTION_INTEGRATION_CONFIG = IntegrationConfig(
    filter_input=True,
    filter_output=True,
    filter_streaming=True,
    safety_level=SafetyLevel.BALANCED,  # Balanced safety for production
    max_retry_attempts=5,  # 5 retry attempts for resilience
    timeout_seconds=30,    # 30 second timeout
    enable_metrics=True,   # Enable comprehensive metrics
    log_all_requests=False,  # Disable verbose logging in production
    block_on_filter=True,  # Block unsafe content
)

# Production Monitoring Configuration  
PRODUCTION_MONITORING_CONFIG = MonitoringConfig(
    enable_metrics=True,
    enable_alerts=True,
    metrics_retention_hours=168,  # 7 days retention
    alert_cooldown_minutes=15,    # 15 minute alert cooldown
    
    # Performance Thresholds
    max_response_time_ms=100,     # Alert if >100ms response time
    max_error_rate=0.05,          # Alert if >5% error rate
    max_memory_usage=0.85,        # Alert if >85% memory usage
    max_cpu_usage=0.80,           # Alert if >80% CPU usage
    
    # Alert Endpoints
    alert_webhook_url=os.getenv("ALERT_WEBHOOK_URL"),
    slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
    email_alerts_enabled=True,
    email_recipients=os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(","),
)

# Environment-specific configurations
ENVIRONMENTS = {
    "development": {
        "optimization": OptimizationConfig(
            cache_size_mb=50,
            max_concurrent_requests=10,
            max_worker_threads=2,
            performance_target_ms=200.0,
        ),
        "safety_level": SafetyLevel.PERMISSIVE,
        "log_level": "DEBUG",
        "metrics_enabled": True,
    },
    
    "staging": {
        "optimization": OptimizationConfig(
            cache_size_mb=100,
            max_concurrent_requests=100,
            max_worker_threads=4,
            performance_target_ms=100.0,
        ),
        "safety_level": SafetyLevel.BALANCED,
        "log_level": "INFO",
        "metrics_enabled": True,
    },
    
    "production": {
        "optimization": PRODUCTION_OPTIMIZATION_CONFIG,
        "integration": PRODUCTION_INTEGRATION_CONFIG,
        "monitoring": PRODUCTION_MONITORING_CONFIG,
        "safety_level": SafetyLevel.BALANCED,
        "log_level": "WARNING",
        "metrics_enabled": True,
    }
}

# Deployment Health Checks
HEALTH_CHECK_CONFIG = {
    "startup_timeout_seconds": 60,
    "readiness_timeout_seconds": 30,
    "liveness_timeout_seconds": 5,
    "health_check_interval_seconds": 10,
    
    # Health check endpoints
    "health_checks": [
        {"name": "cache", "critical": True},
        {"name": "processor", "critical": True},
        {"name": "monitoring", "critical": False},
        {"name": "database", "critical": True},
        {"name": "external_apis", "critical": False},
    ]
}

# Resource Limits
RESOURCE_LIMITS = {
    "memory": {
        "request": "512Mi",
        "limit": "2Gi"
    },
    "cpu": {
        "request": "500m", 
        "limit": "2000m"
    },
    "storage": {
        "cache": "1Gi",
        "logs": "500Mi"
    }
}

# Security Configuration
SECURITY_CONFIG = {
    "enable_rate_limiting": True,
    "rate_limit_requests_per_minute": 1000,
    "enable_api_key_auth": True,
    "enable_request_logging": True,
    "enable_content_validation": True,
    "max_request_size_mb": 10,
    "allowed_content_types": [
        "application/json",
        "text/plain",
        "text/html"
    ]
}

def get_config(environment: str = "production") -> dict:
    """Get configuration for specified environment."""
    if environment not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {environment}")
    
    config = ENVIRONMENTS[environment].copy()
    config.update({
        "environment": environment,
        "health_checks": HEALTH_CHECK_CONFIG,
        "resources": RESOURCE_LIMITS,
        "security": SECURITY_CONFIG,
    })
    
    return config

def validate_production_config() -> bool:
    """Validate production configuration is complete."""
    required_env_vars = [
        "ALERT_WEBHOOK_URL",
        "DATABASE_URL", 
        "REDIS_URL",
        "API_KEY_SECRET"
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Missing required environment variables: {missing_vars}")
        return False
    
    return True

# Deployment Commands
DEPLOYMENT_COMMANDS = {
    "docker_build": "docker build -t cot-safepath-filter:latest .",
    "docker_push": "docker push cot-safepath-filter:latest",
    "kubernetes_deploy": "kubectl apply -f k8s/",
    "health_check": "curl -f http://localhost:8000/health",
    "load_test": "ab -n 1000 -c 10 http://localhost:8000/filter",
}

if __name__ == "__main__":
    # Configuration validation
    print("=== CoT SafePath Filter Production Configuration ===")
    print(f"Environment: {PRODUCTION_CONFIG['environment']}")
    print(f"Version: {PRODUCTION_CONFIG['version']}")
    
    if validate_production_config():
        print("✅ Production configuration is valid")
    else:
        print("❌ Production configuration has issues")
    
    config = get_config("production")
    print(f"📊 Performance target: {config['optimization'].performance_target_ms}ms")
    print(f"🚀 Max concurrent requests: {config['optimization'].max_concurrent_requests}")
    print(f"💾 Cache size: {config['optimization'].cache_size_mb}MB")
    print(f"🔧 Worker threads: {config['optimization'].max_worker_threads}")