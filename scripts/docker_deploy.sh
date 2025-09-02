#!/bin/bash
# Factor Forecasting System - Docker Deployment Script
# Automated deployment with Docker and Docker Compose

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_TYPE=${1:-"development"}  # development, production, monitoring
IMAGE_TAG=${2:-"latest"}
DATA_DIR=${3:-"./data"}
CONFIG_FILE=${4:-"config.yaml"}

# Docker settings
DOCKER_REGISTRY=${DOCKER_REGISTRY:-""}
IMAGE_NAME="factor-forecasting"
FULL_IMAGE_NAME="${DOCKER_REGISTRY}${IMAGE_NAME}:${IMAGE_TAG}"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to validate configuration
validate_configuration() {
    print_status "Validating configuration..."
    
    # Check if data directory exists
    if [ ! -d "$DATA_DIR" ]; then
        print_warning "Data directory $DATA_DIR does not exist. Creating it..."
        mkdir -p "$DATA_DIR"
    fi
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        print_warning "Config file $CONFIG_FILE does not exist. Using default..."
        if [ -f "configs/training_configs/local/local_training.yaml" ]; then
            cp "configs/training_configs/local/local_training.yaml" "$CONFIG_FILE"
            print_status "Created default config file: $CONFIG_FILE"
        else
            print_error "No default config file found. Please create $CONFIG_FILE"
            exit 1
        fi
    fi
    
    print_success "Configuration validation passed"
}

# Function to build Docker image
build_image() {
    print_status "Building Docker image: $FULL_IMAGE_NAME"
    
    # Build with build args for optimization
    docker build \
        --tag "$FULL_IMAGE_NAME" \
        --build-arg PYTHON_VERSION=3.10 \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
        --build-arg VERSION="$IMAGE_TAG" \
        .
    
    print_success "Docker image built successfully"
}

# Function to run security scan (optional)
security_scan() {
    if command -v trivy &> /dev/null; then
        print_status "Running security scan with Trivy..."
        trivy image --severity HIGH,CRITICAL "$FULL_IMAGE_NAME" || print_warning "Security scan found issues"
    else
        print_warning "Trivy not found. Skipping security scan. Install with: curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin"
    fi
}

# Function to deploy with Docker Compose
deploy_services() {
    print_status "Deploying services for $DEPLOYMENT_TYPE environment..."
    
    case $DEPLOYMENT_TYPE in
        "development")
            COMPOSE_FILE="docker-compose.yml"
            ;;
        "production")
            COMPOSE_FILE="deploy/docker-compose.production.yml"
            ;;
        "monitoring")
            COMPOSE_FILE="docker-compose.yml"
            COMPOSE_PROFILES="--profile monitoring"
            ;;
        *)
            print_error "Unknown deployment type: $DEPLOYMENT_TYPE"
            print_status "Available types: development, production, monitoring"
            exit 1
            ;;
    esac
    
    # Set environment variables
    export FACTOR_IMAGE="$FULL_IMAGE_NAME"
    export FACTOR_DATA_DIR="$(realpath $DATA_DIR)"
    export FACTOR_CONFIG_FILE="$(realpath $CONFIG_FILE)"
    
    # Deploy services
    if [ -f "$COMPOSE_FILE" ]; then
        print_status "Using compose file: $COMPOSE_FILE"
        
        # Stop existing services
        docker-compose -f "$COMPOSE_FILE" down --remove-orphans || true
        
        # Start services
        docker-compose -f "$COMPOSE_FILE" up -d $COMPOSE_PROFILES
        
        print_success "Services deployed successfully"
    else
        print_error "Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
}

# Function to wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for API service
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f http://localhost:8000/health &> /dev/null; then
            print_success "API service is ready"
            break
        fi
        
        attempt=$((attempt + 1))
        print_status "Waiting for API service... ($attempt/$max_attempts)"
        sleep 10
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_error "API service failed to start within expected time"
        print_status "Checking service logs..."
        docker-compose logs factor-api
        exit 1
    fi
}

# Function to run health checks
run_health_checks() {
    print_status "Running health checks..."
    
    # Check API health
    if curl -s -f http://localhost:8000/health | grep -q "healthy"; then
        print_success "API health check passed"
    else
        print_error "API health check failed"
        exit 1
    fi
    
    # Check Redis (if available)
    if docker-compose ps | grep -q redis; then
        if docker exec factor-redis redis-cli ping | grep -q PONG; then
            print_success "Redis health check passed"
        else
            print_warning "Redis health check failed"
        fi
    fi
    
    print_success "Health checks completed"
}

# Function to show deployment info
show_deployment_info() {
    echo ""
    echo "========================================"
    echo "Deployment Complete!"
    echo "========================================"
    echo ""
    print_success "Factor Forecasting System has been deployed successfully"
    echo ""
    echo "Deployment Type: $DEPLOYMENT_TYPE"
    echo "Image: $FULL_IMAGE_NAME"
    echo "Data Directory: $DATA_DIR"
    echo "Config File: $CONFIG_FILE"
    echo ""
    echo "Service URLs:"
    echo "  - API Server: http://localhost:8000"
    echo "  - Health Check: http://localhost:8000/health"
    echo "  - API Documentation: http://localhost:8000/docs"
    
    if [ "$DEPLOYMENT_TYPE" = "monitoring" ] || [ "$DEPLOYMENT_TYPE" = "production" ]; then
        echo "  - Grafana Dashboard: http://localhost:3000 (admin/admin)"
        echo "  - Prometheus: http://localhost:9090"
    fi
    
    echo ""
    echo "Docker Commands:"
    echo "  - View logs: docker-compose logs -f"
    echo "  - Stop services: docker-compose down"
    echo "  - Restart services: docker-compose restart"
    echo "  - Scale API: docker-compose up -d --scale factor-api=3"
    echo ""
    echo "Management Commands:"
    echo "  - Train model: docker exec factor-api python -m src.training.train"
    echo "  - Run evaluation: docker exec factor-api python -m src.evaluation.evaluate"
    echo "  - Access shell: docker exec -it factor-api /bin/bash"
    echo ""
}

# Function to cleanup on failure
cleanup_on_failure() {
    print_error "Deployment failed. Cleaning up..."
    docker-compose down --remove-orphans || true
    exit 1
}

# Main deployment flow
main() {
    echo "========================================"
    echo "Factor Forecasting System - Docker Deployment"
    echo "========================================"
    echo ""
    echo "Deployment Type: $DEPLOYMENT_TYPE"
    echo "Image Tag: $IMAGE_TAG"
    echo "Data Directory: $DATA_DIR"
    echo "Config File: $CONFIG_FILE"
    echo ""
    
    # Set up error handling
    trap cleanup_on_failure ERR
    
    check_prerequisites
    validate_configuration
    build_image
    security_scan
    deploy_services
    wait_for_services
    run_health_checks
    show_deployment_info
}

# Handle script arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Factor Forecasting System - Docker Deployment Script"
    echo ""
    echo "Usage: $0 [DEPLOYMENT_TYPE] [IMAGE_TAG] [DATA_DIR] [CONFIG_FILE]"
    echo ""
    echo "Arguments:"
    echo "  DEPLOYMENT_TYPE   Deployment environment (default: development)"
    echo "                    Options: development, production, monitoring"
    echo "  IMAGE_TAG         Docker image tag (default: latest)"
    echo "  DATA_DIR          Data directory path (default: ./data)"
    echo "  CONFIG_FILE       Configuration file path (default: config.yaml)"
    echo ""
    echo "Environment Variables:"
    echo "  DOCKER_REGISTRY   Docker registry URL (optional)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Development deployment"
    echo "  $0 production v2.0.0                 # Production deployment with specific tag"
    echo "  $0 monitoring latest /data/prod      # Monitoring deployment with custom data path"
    echo ""
    echo "Quick Commands:"
    echo "  $0 development                        # Start development environment"
    echo "  $0 production                         # Start production environment"
    echo "  $0 monitoring                         # Start with monitoring stack"
    echo ""
    exit 0
fi

# Run main deployment
main

print_success "Docker deployment completed successfully!"
