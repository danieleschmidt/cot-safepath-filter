#!/bin/bash
set -e

# CoT SafePath Filter Production Entrypoint
echo "=== CoT SafePath Filter Starting ==="
echo "Environment: ${ENVIRONMENT:-production}"
echo "Version: $(python -c 'import cot_safepath; print(cot_safepath.__version__)')"
echo "Workers: ${WORKERS:-4}"
echo "Port: ${PORT:-8000}"

# Validate environment
echo "🔍 Validating environment configuration..."

# Check required environment variables
REQUIRED_VARS=("DATABASE_URL" "REDIS_URL")
for var in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var}" ]]; then
        echo "❌ Error: Required environment variable $var is not set"
        exit 1
    fi
done

echo "✅ Environment validation passed"

# Run pre-flight checks
echo "🚀 Running pre-flight checks..."

# Test Python import
python -c "
import cot_safepath
from cot_safepath import SafePathFilter, HighPerformanceFilterEngine
from cot_safepath.performance_optimization import OptimizationConfig

# Test basic functionality
config = OptimizationConfig()
engine = HighPerformanceFilterEngine(config)
print('✅ Core modules loaded successfully')
"

# Validate configuration
python -c "
from deployment.production_config import validate_production_config, get_config
if validate_production_config():
    print('✅ Production configuration validated')
    config = get_config('${ENVIRONMENT:-production}')
    print(f'📊 Performance target: {config[\"optimization\"].performance_target_ms}ms')
else:
    print('❌ Configuration validation failed')
    exit(1)
"

# Database connectivity check
echo "🔍 Testing database connectivity..."
python -c "
import os
import redis
import asyncio
from sqlalchemy import create_engine, text

# Test Redis connection
redis_client = redis.from_url(os.getenv('REDIS_URL'))
redis_client.ping()
print('✅ Redis connection successful')

# Test database connection
engine = create_engine(os.getenv('DATABASE_URL'))
with engine.connect() as conn:
    conn.execute(text('SELECT 1'))
print('✅ Database connection successful')
"

# Warm up caches
echo "🔥 Warming up application caches..."
python -c "
from cot_safepath.performance_optimization import HighPerformanceFilterEngine, OptimizationConfig
from cot_safepath.models import FilterRequest, SafetyLevel

config = OptimizationConfig(cache_size_mb=50)
engine = HighPerformanceFilterEngine(config)

# Warm up with test requests
test_requests = [
    FilterRequest(content='test content', safety_level=SafetyLevel.BALANCED)
    for _ in range(10)
]

print('✅ Cache warmed up successfully')
"

# Start monitoring
echo "📊 Initializing monitoring..."
python -c "
from cot_safepath.realtime_monitoring import get_global_monitor, configure_monitoring
from deployment.production_config import PRODUCTION_MONITORING_CONFIG

configure_monitoring(PRODUCTION_MONITORING_CONFIG)
monitor = get_global_monitor()
print('✅ Monitoring initialized')
"

echo "🎉 Pre-flight checks completed successfully"
echo "🚀 Starting CoT SafePath Filter application..."

# Handle graceful shutdown
trap 'echo "🛑 Received shutdown signal, gracefully stopping..."; kill -TERM $PID; wait $PID' TERM INT

# Execute the main command
exec "$@" &
PID=$!
wait $PID