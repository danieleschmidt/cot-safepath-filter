#!/bin/bash
# Build script for CoT SafePath Filter
# Handles Docker builds, testing, and packaging

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="cot-safepath-filter"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
VERSION=${VERSION:-$(grep '^version = ' pyproject.toml | cut -d'"' -f2)}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Build script for CoT SafePath Filter

Usage: $0 [OPTIONS] COMMAND

Commands:
    build           Build Docker images
    test            Run tests
    package         Build Python packages
    all             Run build, test, and package
    clean           Clean build artifacts
    push            Push Docker images to registry
    security-scan   Run security scans

Options:
    -h, --help      Show this help message
    -v, --version   Show version information
    --no-cache      Build without Docker cache
    --target TARGET Specify build target (production, development)
    --platform PLATFORM Specify target platform
    --push          Push images after building
    --registry REGISTRY Specify Docker registry

Environment Variables:
    VERSION         Version tag for images (default: from pyproject.toml)
    REGISTRY        Docker registry URL
    BUILD_ARGS      Additional build arguments

Examples:
    $0 build                    # Build production image
    $0 build --target development  # Build development image
    $0 test                     # Run test suite
    $0 all --push              # Build, test, package, and push
    $0 security-scan            # Run security scans

EOF
}

# Version function
show_version() {
    echo "CoT SafePath Filter Build Script v1.0.0"
    echo "Project Version: $VERSION"
    echo "Build Date: $BUILD_DATE"
    echo "VCS Ref: $VCS_REF"
}

# Prerequisites check
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
    command -v git >/dev/null 2>&1 || missing_tools+=("git")
    command -v python3 >/dev/null 2>&1 || missing_tools+=("python3")
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker images
build_images() {
    local target=${1:-production}
    local no_cache=${2:-false}
    local platform=${3:-}
    
    log_info "Building Docker images..."
    log_info "Target: $target"
    log_info "Version: $VERSION"
    log_info "Build Date: $BUILD_DATE"
    log_info "VCS Ref: $VCS_REF"
    
    local build_args=(
        --build-arg "BUILD_DATE=$BUILD_DATE"
        --build-arg "VERSION=$VERSION"
        --build-arg "VCS_REF=$VCS_REF"
        --target "$target"
        --tag "$IMAGE_NAME:$VERSION"
        --tag "$IMAGE_NAME:latest"
    )
    
    if [ "$no_cache" = true ]; then
        build_args+=(--no-cache)
    fi
    
    if [ -n "$platform" ]; then
        build_args+=(--platform "$platform")
    fi
    
    # Add registry prefix if specified
    if [ -n "${REGISTRY:-}" ]; then
        build_args+=(--tag "$REGISTRY/$IMAGE_NAME:$VERSION")
        build_args+=(--tag "$REGISTRY/$IMAGE_NAME:latest")
    fi
    
    cd "$PROJECT_ROOT"
    
    if docker build "${build_args[@]}" .; then
        log_success "Docker build completed successfully"
        
        # Show image info
        echo ""
        docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    else
        log_error "Docker build failed"
        exit 1
    fi
}

# Run tests
run_tests() {
    log_info "Running test suite..."
    
    cd "$PROJECT_ROOT"
    
    # Build test image
    if docker build --target development -t "$IMAGE_NAME:test" .; then
        log_success "Test image built successfully"
    else
        log_error "Failed to build test image"
        exit 1
    fi
    
    # Run tests in container
    local test_cmd=(
        docker run --rm
        -v "$PROJECT_ROOT:/workspace"
        -w /workspace
        -e ENVIRONMENT=test
        "$IMAGE_NAME:test"
        python -m pytest tests/ -v --cov=src/cot_safepath --cov-report=term-missing --cov-report=html
    )
    
    if "${test_cmd[@]}"; then
        log_success "All tests passed"
    else
        log_error "Tests failed"
        exit 1
    fi
}

# Build Python packages
build_packages() {
    log_info "Building Python packages..."
    
    cd "$PROJECT_ROOT"
    
    # Clean previous builds
    rm -rf build/ dist/ *.egg-info/
    
    # Build packages
    if python -m build; then
        log_success "Python packages built successfully"
        
        # Show package info
        echo ""
        ls -la dist/
    else
        log_error "Package build failed"
        exit 1
    fi
}

# Security scanning
security_scan() {
    log_info "Running security scans..."
    
    cd "$PROJECT_ROOT"
    
    # Trivy scan
    if command -v trivy >/dev/null 2>&1; then
        log_info "Running Trivy security scan..."
        trivy image "$IMAGE_NAME:latest"
    else
        log_warning "Trivy not installed, skipping container scan"
    fi
    
    # Bandit scan
    log_info "Running Bandit security scan..."
    if docker run --rm -v "$PROJECT_ROOT:/workspace" -w /workspace "$IMAGE_NAME:test" \
        python -m bandit -r src/ -f json -o bandit-report.json; then
        log_success "Bandit scan completed"
    else
        log_warning "Bandit scan found issues"
    fi
    
    # Safety scan
    log_info "Running Safety dependency scan..."
    if docker run --rm -v "$PROJECT_ROOT:/workspace" -w /workspace "$IMAGE_NAME:test" \
        python -m safety check --json --output safety-report.json; then
        log_success "Safety scan completed"
    else
        log_warning "Safety scan found vulnerabilities"
    fi
}

# Push images
push_images() {
    if [ -z "${REGISTRY:-}" ]; then
        log_error "REGISTRY environment variable not set"
        exit 1
    fi
    
    log_info "Pushing images to $REGISTRY..."
    
    local images=(
        "$REGISTRY/$IMAGE_NAME:$VERSION"
        "$REGISTRY/$IMAGE_NAME:latest"
    )
    
    for image in "${images[@]}"; do
        if docker push "$image"; then
            log_success "Pushed $image"
        else
            log_error "Failed to push $image"
            exit 1
        fi
    done
}

# Clean build artifacts
clean_build() {
    log_info "Cleaning build artifacts..."
    
    cd "$PROJECT_ROOT"
    
    # Clean Python artifacts
    rm -rf build/ dist/ *.egg-info/
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    rm -rf .pytest_cache/ .coverage htmlcov/ .mypy_cache/ .ruff_cache/
    
    # Clean Docker artifacts
    docker image prune -f 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Main function
main() {
    local command=""
    local target="production"
    local no_cache=false
    local platform=""
    local push=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--version)
                show_version
                exit 0
                ;;
            --no-cache)
                no_cache=true
                shift
                ;;
            --target)
                target="$2"
                shift 2
                ;;
            --platform)
                platform="$2"
                shift 2
                ;;
            --push)
                push=true
                shift
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            build|test|package|all|clean|push|security-scan)
                command="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    if [ -z "$command" ]; then
        log_error "No command specified"
        show_help
        exit 1
    fi
    
    # Execute command
    case $command in
        build)
            check_prerequisites
            build_images "$target" "$no_cache" "$platform"
            if [ "$push" = true ]; then
                push_images
            fi
            ;;
        test)
            check_prerequisites
            run_tests
            ;;
        package)
            check_prerequisites
            build_packages
            ;;
        all)
            check_prerequisites
            build_images "$target" "$no_cache" "$platform"
            run_tests
            build_packages
            if [ "$push" = true ]; then
                push_images
            fi
            ;;
        clean)
            clean_build
            ;;
        push)
            push_images
            ;;
        security-scan)
            check_prerequisites
            security_scan
            ;;
        *)
            log_error "Unknown command: $command"
            exit 1
            ;;
    esac
    
    log_success "Build script completed successfully"
}

# Run main function with all arguments
main "$@"