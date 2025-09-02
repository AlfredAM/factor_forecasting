#!/bin/bash

# Factor Forecasting Unified Scripts
# Combines all script functionality into one file

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_info() {
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

# Check Python environment
check_python_env() {
    print_info "Checking Python environment..."
    
    if ! command -v python &> /dev/null; then
        print_error "Python not installed"
        exit 1
    fi
    
    if ! python -c "import torch" &> /dev/null; then
        print_warning "PyTorch not installed, please install dependencies first"
        exit 1
    fi
    
    print_success "Python environment check passed"
}

# Set up environment variables
setup_environment() {
    print_info "Setting up environment variables..."
    
    # Get script directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
    
    print_success "Environment variables set: PYTHONPATH=$PYTHONPATH"
}

# Check data files
check_data_files() {
    print_info "Checking data files..."
    
    if [ ! -f "data/raw/sample_50k.parquet" ]; then
        print_warning "sample_50k.parquet not found, will use synthetic data"
    else
        print_success "Found data file: data/raw/sample_50k.parquet"
    fi
}

# Quick setup function
quick_setup() {
    print_info "Running quick setup..."
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    print_info "Activating virtual environment..."
    source venv/bin/activate

    # Install dependencies
    print_info "Installing dependencies..."
    pip install -r configs/requirements.txt

    # Create necessary directories
    print_info "Creating directories..."
    mkdir -p models logs outputs/{checkpoints,results,plots} data/{raw,processed}

    # Check if data exists
    if [ ! -f "data/raw/20180102.parquet" ]; then
        print_warning "No data files found in data/raw/"
        print_info "Please place your data files in data/raw/ directory"
    fi

    print_success "Quick setup completed!"
}

# Local training
run_local_training() {
    print_info "Starting local training..."
    
    python src/training/rolling_train.py 2>&1 | tee rolling_train.log
    
    if [ $? -eq 0 ]; then
        print_success "Local training completed"
    else
        print_error "Local training failed"
        exit 1
    fi
}

# Server training
run_server_training() {
    local server_ip=$1
    
    print_info "Connecting to server: $server_ip"
    
    if [ ! -f "connect_server.exp" ]; then
        print_error "connect_server.exp script not found"
        exit 1
    fi
    
    # Sync code to server
    print_info "Syncing code to server..."
    ./connect_server.exp "$server_ip" "cd /home/ecs-user/deployment_package && python src/training/rolling_train.py"
}

# Manual restart instructions
show_manual_restart() {
    echo "=========================================="
    echo "Manual Restart Training Instructions"
    echo "=========================================="
    echo ""
    echo "Please follow these steps to manually execute:"
    echo ""
    echo "1. SSH login to server:"
    echo "   ssh ecs-user@47.120.44.48"
    echo "   Password: Abab1234"
    echo ""
    echo "2. Navigate to project directory:"
    echo "   cd /home/ecs-user/deployment_package"
    echo ""
    echo "3. Activate miniconda environment:"
    echo "   source ~/miniconda3/bin/activate"
    echo ""
    echo "4. Activate factor_forecast environment:"
    echo "   conda activate factor_forecast"
    echo ""
    echo "5. Check data files on server:"
    echo "   ls -la /nas/feature_v2_10s/"
    echo "   ls -la /nas/feature_v2_10s/ | head -20"
    echo "   find /nas/feature_v2_10s/ -name '*.parquet' | head -10"
    echo ""
    echo "6. Start rolling window training:"
    echo "   export PYTHONPATH=/home/ecs-user/deployment_package"
    echo "   python src/training/rolling_train.py"
    echo ""
    echo "Alternative commands:"
    echo "   nohup python src/training/rolling_train.py > training.log 2>&1 &"
    echo "   tail -f training.log"
    echo ""
    echo "To check GPU status:"
    echo "   nvidia-smi"
    echo "   watch -n 1 nvidia-smi"
    echo ""
    echo "To kill training process:"
    echo "   ps aux | grep python"
    echo "   kill -9 <process_id>"
    echo ""
    echo "Manual execution steps completed!"
}

# Show available commands
show_available_commands() {
    echo ""
    echo "Available commands:"
    echo "  python main.py --mode api          # Start API server"
    echo "  python main.py --mode rolling      # Start rolling training"
    echo "  python main.py --mode train        # Start regular training"
    echo "  python main.py --mode test         # Run tests"
    echo ""
    echo "Docker commands:"
    echo "  cd deploy && docker-compose up factor-forecast-api"
    echo "  cd deploy && docker-compose --profile rolling up factor-forecast-rolling"
    echo ""
    echo "Server deployment:"
    echo "  python deploy/deploy_to_aliyun.py"
    echo ""
}

# Show help information
show_help() {
    echo "Factor Forecasting Unified Scripts"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  setup                    Run quick setup (create venv, install deps)"
    echo "  local                    Run training locally"
    echo "  server <ip>              Run training on specified server"
    echo "  check                    Check environment"
    echo "  restart                  Show manual restart instructions"
    echo "  commands                 Show available commands"
    echo "  help                     Show this help information"
    echo ""
    echo "Examples:"
    echo "  $0 setup                 # Quick setup"
    echo "  $0 local                 # Local training"
    echo "  $0 server 47.120.44.48   # Server training"
    echo "  $0 check                 # Environment check"
    echo "  $0 restart               # Show restart instructions"
}

# Main function
main() {
    case "${1:-help}" in
        "setup")
            quick_setup
            ;;
        "local")
            check_python_env
            setup_environment
            check_data_files
            run_local_training
            ;;
        "server")
            if [ -z "$2" ]; then
                print_error "Please specify server IP address"
                show_help
                exit 1
            fi
            check_python_env
            setup_environment
            run_server_training "$2"
            ;;
        "check")
            check_python_env
            setup_environment
            check_data_files
            print_success "Environment check completed"
            ;;
        "restart")
            show_manual_restart
            ;;
        "commands")
            show_available_commands
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function
main "$@" 