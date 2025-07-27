#!/bin/bash

# Docker entrypoint script for CoT SafePath Filter
# Handles initialization, configuration, and graceful startup

set -e

# Default values
DEFAULT_PORT=8080
DEFAULT_WORKERS=1
DEFAULT_LOG_LEVEL=INFO

# Environment variables with defaults
PORT=${PORT:-$DEFAULT_PORT}
WORKERS=${WORKERS:-$DEFAULT_WORKERS}
LOG_LEVEL=${LOG_LEVEL:-$DEFAULT_LOG_LEVEL}
ENVIRONMENT=${ENVIRONMENT:-production}

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [[ "$LOG_LEVEL" == "DEBUG" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Function to check if a service is available
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-30}
    
    log_info "Waiting for $service_name at $host:$port..."
    
    for i in $(seq 1 $timeout); do
        if nc -z "$host" "$port" 2>/dev/null; then
            log_info "$service_name is available"
            return 0
        fi
        sleep 1
    done
    
    log_error "$service_name is not available after ${timeout}s"
    return 1
}

# Function to validate environment
validate_environment() {
    log_info "Validating environment configuration..."
    
    # Check required environment variables
    if [[ -z "$DATABASE_URL" ]] && [[ "$ENVIRONMENT" == "production" ]]; then
        log_warn "DATABASE_URL not set, using SQLite"
        export DATABASE_URL="sqlite:///./safepath.db"
    fi
    
    # Validate port
    if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [[ "$PORT" -lt 1 ]] || [[ "$PORT" -gt 65535 ]]; then
        log_error "Invalid port: $PORT"
        exit 1
    fi
    
    # Validate workers
    if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [[ "$WORKERS" -lt 1 ]]; then
        log_error "Invalid workers count: $WORKERS"
        exit 1
    fi
    
    # Validate log level
    case "$LOG_LEVEL" in
        DEBUG|INFO|WARNING|ERROR|CRITICAL)
            ;;
        *)
            log_error "Invalid log level: $LOG_LEVEL"
            exit 1
            ;;
    esac
    
    log_info "Environment validation completed"
}

# Function to initialize application
initialize_app() {
    log_info "Initializing CoT SafePath Filter..."
    
    # Create necessary directories
    mkdir -p /app/logs /app/data /app/tmp
    
    # Initialize database if needed
    if [[ "$DATABASE_URL" == sqlite* ]]; then
        log_info "Initializing SQLite database..."
        # Run database migrations if available
        if command -v alembic &> /dev/null; then
            alembic upgrade head || log_warn "Database migration failed or not needed"
        fi
    fi
    
    # Wait for external services
    if [[ -n "$REDIS_URL" ]]; then
        # Extract host and port from Redis URL
        redis_host=$(echo "$REDIS_URL" | sed -E 's|redis://([^:]+):([0-9]+).*|\1|')
        redis_port=$(echo "$REDIS_URL" | sed -E 's|redis://([^:]+):([0-9]+).*|\2|')
        if [[ -n "$redis_host" ]] && [[ -n "$redis_port" ]]; then
            wait_for_service "$redis_host" "$redis_port" "Redis" 30 || log_warn "Redis not available, continuing without cache"
        fi
    fi
    
    log_info "Application initialization completed"
}

# Function to run health check
health_check() {
    log_info "Running application health check..."
    
    # Basic Python import test
    python -c "import cot_safepath; print('Package import successful')" || {
        log_error "Package import failed"
        exit 1
    }
    
    # Configuration validation
    python -c "
from cot_safepath.config import validate_config
try:
    validate_config()
    print('Configuration validation successful')
except Exception as e:
    print(f'Configuration validation failed: {e}')
    exit(1)
" || {
        log_error "Configuration validation failed"
        exit 1
    }
    
    log_info "Health check completed successfully"
}

# Function to start the server
start_server() {
    log_info "Starting CoT SafePath Filter server..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Port: $PORT"
    log_info "Workers: $WORKERS"
    log_info "Log Level: $LOG_LEVEL"
    
    if [[ "$ENVIRONMENT" == "development" ]]; then
        # Development mode with hot reload
        exec uvicorn cot_safepath.server:app \
            --host 0.0.0.0 \
            --port "$PORT" \
            --log-level "$(echo "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')" \
            --reload \
            --reload-dir /app/src
    else
        # Production mode
        exec uvicorn cot_safepath.server:app \
            --host 0.0.0.0 \
            --port "$PORT" \
            --workers "$WORKERS" \
            --log-level "$(echo "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')" \
            --access-log \
            --loop uvloop \
            --http httptools
    fi
}

# Function to run CLI command
run_cli() {
    log_info "Running CLI command: $*"
    exec safepath "$@"
}

# Function to run monitoring
run_monitor() {
    log_info "Starting monitoring dashboard..."
    exec safepath-monitor \
        --port "${MONITOR_PORT:-9090}" \
        --log-level "$LOG_LEVEL"
}

# Signal handlers for graceful shutdown
shutdown() {
    log_info "Received shutdown signal, gracefully stopping..."
    # Add any cleanup logic here
    exit 0
}

# Trap signals
trap shutdown SIGTERM SIGINT

# Main execution logic
main() {
    log_info "CoT SafePath Filter Docker Container Starting..."
    log_info "Version: ${VERSION:-unknown}"
    log_info "Build Date: ${BUILD_DATE:-unknown}"
    
    # Validate environment first
    validate_environment
    
    # Initialize application
    initialize_app
    
    # Run health check
    health_check
    
    # Determine what to run based on command
    case "${1:-safepath-server}" in
        safepath-server)
            start_server
            ;;
        safepath-monitor)
            run_monitor
            ;;
        safepath)
            shift
            run_cli "$@"
            ;;
        bash|sh)
            log_info "Starting interactive shell..."
            exec "$@"
            ;;
        health-check)
            log_info "Running health check only..."
            health_check
            log_info "Health check passed, exiting"
            ;;
        *)
            log_info "Running custom command: $*"
            exec "$@"
            ;;
    esac
}

# Run main function with all arguments
main "$@"