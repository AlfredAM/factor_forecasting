#!/bin/bash

# Factor Forecasting Docker Services Startup Script
set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_docker() {
    log_info "Checking Docker status..."
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    log_success "Docker is running"
}

create_directories() {
    log_info "Creating necessary directories..."
    mkdir -p ../models ../logs ../outputs ../data ../checkpoints
    log_success "Directories created"
}

start_services() {
    case "${1:-all}" in
        "basic")
            log_info "Starting basic services..."
            docker-compose up -d factor-forecast-api factor-forecast-dev
            ;;
        "training")
            log_info "Starting training services..."
            docker-compose --profile train up -d
            ;;
        "rolling")
            log_info "Starting rolling training services..."
            docker-compose --profile rolling up -d
            ;;
        "all")
            log_info "Starting all services..."
            docker-compose --profile train --profile rolling --profile monitoring --profile production up -d
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
    log_success "Services started"
}

main() {
    echo "Factor Forecasting Services Startup Script"
    check_docker
    create_directories
    start_services "$1"
    
    echo ""
    log_success "Services startup completed!"
    echo "Use 'docker-compose ps' to check status"
    echo "Use 'docker-compose logs -f' to view logs"
}

main "$@"
