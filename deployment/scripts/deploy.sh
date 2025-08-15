#!/bin/bash
set -euo pipefail

# SafePath Filter Deployment Script
# Automated deployment script for production environments

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_DIR="${PROJECT_ROOT}/deployment"

# Default values
ENVIRONMENT="${ENVIRONMENT:-production}"
AWS_REGION="${AWS_REGION:-us-west-2}"
CLUSTER_NAME="${CLUSTER_NAME:-safepath-filter}"
NAMESPACE="${NAMESPACE:-safepath}"
CHART_VERSION="${CHART_VERSION:-1.0.0}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_TERRAFORM="${SKIP_TERRAFORM:-false}"
SKIP_BUILD="${SKIP_BUILD:-false}"

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

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Error handling
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Deployment failed with exit code $exit_code"
        log_info "Check logs above for detailed error information"
    fi
    exit $exit_code
}
trap cleanup EXIT

# Utility functions
check_dependencies() {
    log_info "Checking dependencies..."
    
    local deps=("aws" "kubectl" "helm" "terraform" "docker")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Please install the missing dependencies and try again"
        exit 1
    fi
    
    log_success "All dependencies found"
}

validate_aws_credentials() {
    log_info "Validating AWS credentials..."
    
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured or invalid"
        log_info "Please run 'aws configure' or set AWS environment variables"
        exit 1
    fi
    
    local identity=$(aws sts get-caller-identity --output text --query 'Arn')
    log_success "AWS credentials validated: $identity"
}

build_docker_image() {
    if [ "$SKIP_BUILD" = "true" ]; then
        log_info "Skipping Docker image build (SKIP_BUILD=true)"
        return
    fi
    
    log_info "Building Docker image..."
    
    local image_name="terragonlabs/cot-safepath-filter:${IMAGE_TAG}"
    local dockerfile="${DEPLOYMENT_DIR}/docker/Dockerfile.production"
    
    if [ ! -f "$dockerfile" ]; then
        log_error "Dockerfile not found: $dockerfile"
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
    
    # Build image
    log_info "Building image: $image_name"
    docker build -f "$dockerfile" -t "$image_name" .
    
    # Push to registry (if not dry run)
    if [ "$DRY_RUN" != "true" ]; then
        log_info "Pushing image to registry..."
        docker push "$image_name"
        log_success "Image pushed successfully"
    else
        log_info "DRY RUN: Would push image $image_name"
    fi
}

deploy_infrastructure() {
    if [ "$SKIP_TERRAFORM" = "true" ]; then
        log_info "Skipping infrastructure deployment (SKIP_TERRAFORM=true)"
        return
    fi
    
    log_info "Deploying infrastructure with Terraform..."
    
    local terraform_dir="${DEPLOYMENT_DIR}/terraform"
    
    if [ ! -d "$terraform_dir" ]; then
        log_error "Terraform directory not found: $terraform_dir"
        exit 1
    fi
    
    cd "$terraform_dir"
    
    # Initialize Terraform
    log_info "Initializing Terraform..."
    terraform init
    
    # Validate configuration
    log_info "Validating Terraform configuration..."
    terraform validate
    
    # Plan deployment
    log_info "Planning infrastructure deployment..."
    terraform plan \
        -var="environment=${ENVIRONMENT}" \
        -var="aws_region=${AWS_REGION}" \
        -var="cluster_name=${CLUSTER_NAME}" \
        -out=tfplan
    
    # Apply (if not dry run)
    if [ "$DRY_RUN" != "true" ]; then
        log_info "Applying infrastructure changes..."
        terraform apply tfplan
        log_success "Infrastructure deployed successfully"
        
        # Update kubeconfig
        log_info "Updating kubeconfig..."
        aws eks update-kubeconfig --region "$AWS_REGION" --name "$CLUSTER_NAME"
        log_success "kubeconfig updated"
    else
        log_info "DRY RUN: Would apply infrastructure changes"
    fi
}

wait_for_cluster() {
    log_info "Waiting for cluster to be ready..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if kubectl cluster-info &> /dev/null; then
            log_success "Cluster is ready"
            return
        fi
        
        log_info "Attempt $attempt/$max_attempts: Cluster not ready, waiting..."
        sleep 10
        ((attempt++))
    done
    
    log_error "Cluster failed to become ready after $max_attempts attempts"
    exit 1
}

create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Namespace $NAMESPACE already exists"
    else
        if [ "$DRY_RUN" != "true" ]; then
            kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/namespace.yaml"
            log_success "Namespace created"
        else
            log_info "DRY RUN: Would create namespace $NAMESPACE"
        fi
    fi
}

deploy_application() {
    log_info "Deploying SafePath Filter application..."
    
    local helm_dir="${DEPLOYMENT_DIR}/helm"
    local values_file="${helm_dir}/values.yaml"
    
    if [ ! -d "$helm_dir" ] || [ ! -f "$values_file" ]; then
        log_error "Helm chart not found in: $helm_dir"
        exit 1
    fi
    
    # Add custom values for this deployment
    local temp_values=$(mktemp)
    cat > "$temp_values" <<EOF
image:
  tag: "${IMAGE_TAG}"

app:
  environment: "${ENVIRONMENT}"

ingress:
  enabled: true
  hosts:
    - host: safepath.terragonlabs.com
      paths:
        - path: /
          pathType: Prefix

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20

monitoring:
  enabled: true

redis:
  enabled: false  # Using external Redis

postgresql:
  enabled: false  # Using external PostgreSQL
EOF
    
    # Deploy with Helm
    if [ "$DRY_RUN" != "true" ]; then
        log_info "Installing/upgrading Helm release..."
        helm upgrade --install safepath-filter "$helm_dir" \
            --namespace "$NAMESPACE" \
            --values "$values_file" \
            --values "$temp_values" \
            --wait \
            --timeout=10m
        
        log_success "Application deployed successfully"
    else
        log_info "DRY RUN: Would deploy application with Helm"
        helm template safepath-filter "$helm_dir" \
            --namespace "$NAMESPACE" \
            --values "$values_file" \
            --values "$temp_values" > /dev/null
        log_info "DRY RUN: Helm template validation passed"
    fi
    
    # Cleanup
    rm -f "$temp_values"
}

verify_deployment() {
    if [ "$DRY_RUN" = "true" ]; then
        log_info "Skipping deployment verification (dry run)"
        return
    fi
    
    log_info "Verifying deployment..."
    
    # Wait for pods to be ready
    log_info "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=cot-safepath-filter -n "$NAMESPACE" --timeout=300s
    
    # Check deployment status
    local ready_replicas=$(kubectl get deployment safepath-filter -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
    local desired_replicas=$(kubectl get deployment safepath-filter -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
    
    if [ "$ready_replicas" = "$desired_replicas" ]; then
        log_success "All $ready_replicas replicas are ready"
    else
        log_error "Only $ready_replicas/$desired_replicas replicas are ready"
        exit 1
    fi
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    local pod_name=$(kubectl get pods -l app.kubernetes.io/name=cot-safepath-filter -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}')
    
    if kubectl exec -n "$NAMESPACE" "$pod_name" -- curl -f http://localhost:8080/health &> /dev/null; then
        log_success "Health endpoint is responding"
    else
        log_error "Health endpoint is not responding"
        exit 1
    fi
    
    # Display service information
    log_info "Service information:"
    kubectl get svc -n "$NAMESPACE"
    
    log_info "Pod information:"
    kubectl get pods -n "$NAMESPACE" -o wide
}

run_tests() {
    if [ "$DRY_RUN" = "true" ]; then
        log_info "Skipping tests (dry run)"
        return
    fi
    
    log_info "Running deployment tests..."
    
    # Run Helm tests
    if helm test safepath-filter -n "$NAMESPACE" --timeout=5m; then
        log_success "Helm tests passed"
    else
        log_error "Helm tests failed"
        exit 1
    fi
    
    # Run custom smoke tests
    log_info "Running smoke tests..."
    local test_script="${SCRIPT_DIR}/smoke-tests.sh"
    
    if [ -f "$test_script" ]; then
        if bash "$test_script" "$NAMESPACE"; then
            log_success "Smoke tests passed"
        else
            log_error "Smoke tests failed"
            exit 1
        fi
    else
        log_warn "Smoke test script not found: $test_script"
    fi
}

display_summary() {
    log_info "Deployment Summary"
    echo "==================="
    echo "Environment: $ENVIRONMENT"
    echo "Region: $AWS_REGION"
    echo "Cluster: $CLUSTER_NAME"
    echo "Namespace: $NAMESPACE"
    echo "Image Tag: $IMAGE_TAG"
    echo "Dry Run: $DRY_RUN"
    echo ""
    
    if [ "$DRY_RUN" != "true" ]; then
        echo "Access URLs:"
        echo "- Application: https://safepath.terragonlabs.com"
        echo "- Monitoring: kubectl port-forward -n $NAMESPACE svc/safepath-filter-service 8080:80"
        echo ""
        echo "Useful Commands:"
        echo "- View logs: kubectl logs -f deployment/safepath-filter -n $NAMESPACE"
        echo "- Port forward: kubectl port-forward -n $NAMESPACE svc/safepath-filter-service 8080:80"
        echo "- Scale deployment: kubectl scale deployment safepath-filter --replicas=5 -n $NAMESPACE"
    fi
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy SafePath Filter to Kubernetes

Options:
    -e, --environment    Environment (dev/staging/production) [default: production]
    -r, --region         AWS region [default: us-west-2]
    -c, --cluster        EKS cluster name [default: safepath-filter]
    -n, --namespace      Kubernetes namespace [default: safepath]
    -t, --tag            Docker image tag [default: latest]
    -d, --dry-run        Perform a dry run without making changes
    --skip-terraform     Skip Terraform infrastructure deployment
    --skip-build         Skip Docker image build
    -h, --help           Show this help message

Environment Variables:
    ENVIRONMENT          Same as --environment
    AWS_REGION           Same as --region
    CLUSTER_NAME         Same as --cluster
    NAMESPACE            Same as --namespace
    IMAGE_TAG            Same as --tag
    DRY_RUN              Same as --dry-run (true/false)
    SKIP_TERRAFORM       Skip Terraform (true/false)
    SKIP_BUILD           Skip build (true/false)

Examples:
    # Deploy to production
    $0 --environment production --region us-west-2

    # Dry run deployment
    $0 --dry-run

    # Deploy only application (skip infrastructure)
    $0 --skip-terraform

    # Deploy with custom image tag
    $0 --tag v1.2.3
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -r|--region)
            AWS_REGION="$2"
            shift 2
            ;;
        -c|--cluster)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        --skip-terraform)
            SKIP_TERRAFORM="true"
            shift
            ;;
        --skip-build)
            SKIP_BUILD="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log_info "Starting SafePath Filter deployment"
    log_info "Environment: $ENVIRONMENT, Region: $AWS_REGION, Cluster: $CLUSTER_NAME"
    
    # Validation
    check_dependencies
    validate_aws_credentials
    
    # Deployment steps
    build_docker_image
    deploy_infrastructure
    wait_for_cluster
    create_namespace
    deploy_application
    verify_deployment
    run_tests
    
    # Summary
    display_summary
    
    log_success "SafePath Filter deployment completed successfully!"
}

# Run main function
main "$@"