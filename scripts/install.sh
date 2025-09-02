#!/bin/bash
# Factor Forecasting System Installation Script
# Professional quantitative finance system setup

set -e  # Exit on any error

echo "========================================"
echo "Factor Forecasting System Installation"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_MIN_VERSION="3.8"
INSTALL_TYPE=${1:-"standard"}  # standard, gpu, dev, api
DATA_DIR=${2:-"./data"}
CONFIG_FILE=${3:-"configs/training_configs/local/local_training.yaml"}

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

# Function to check Python version
check_python_version() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python ${PYTHON_MIN_VERSION}+ first."
        exit 1
    fi
    
    PYTHON_VERSION=$(${PYTHON_CMD} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    
    if [ "$(printf '%s\n' "$PYTHON_MIN_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$PYTHON_MIN_VERSION" ]; then
        print_success "Python ${PYTHON_VERSION} found (minimum ${PYTHON_MIN_VERSION} required)"
    else
        print_error "Python ${PYTHON_VERSION} found, but ${PYTHON_MIN_VERSION}+ is required"
        exit 1
    fi
}

# Function to create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ ! -d "venv" ]; then
        ${PYTHON_CMD} -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_status "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    print_success "Package managers updated"
}

# Function to install dependencies based on type
install_dependencies() {
    print_status "Installing dependencies for ${INSTALL_TYPE} installation..."
    
    case $INSTALL_TYPE in
        "standard")
            pip install -e .
            ;;
        "gpu")
            pip install -e .[gpu]
            ;;
        "dev")
            pip install -e .[dev]
            ;;
        "api")
            pip install -e .[api]
            ;;
        "full")
            pip install -e .[dev,gpu,api,monitoring]
            ;;
        *)
            print_error "Unknown installation type: ${INSTALL_TYPE}"
            print_status "Available types: standard, gpu, dev, api, full"
            exit 1
            ;;
    esac
    
    print_success "Dependencies installed successfully"
}

# Function to setup directories
setup_directories() {
    print_status "Setting up directory structure..."
    
    # Create necessary directories
    mkdir -p "${DATA_DIR}"
    mkdir -p "outputs/models"
    mkdir -p "outputs/logs"
    mkdir -p "outputs/results"
    mkdir -p "outputs/plots"
    
    print_success "Directory structure created"
    
    # Set permissions
    chmod 755 "${DATA_DIR}"
    chmod 755 outputs
    
    print_status "Directory permissions set"
}

# Function to validate installation
validate_installation() {
    print_status "Validating installation..."
    
    # Test imports
    ${PYTHON_CMD} -c "
import sys
sys.path.append('.')

try:
    from src.models.models import create_model
    from src.training.trainer import FactorForecastingTrainer
    from src.data_processing.quantitative_data_processor import QuantitativeDataConfig
    from src.utils.quantitative_metrics import QuantitativePerformanceAnalyzer
    from src.utils.risk_management import RiskManager
    print('SUCCESS: All core modules imported successfully')
except ImportError as e:
    print(f'ERROR: Import failed - {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "Core modules validation passed"
    else
        print_error "Core modules validation failed"
        exit 1
    fi
}

# Function to create sample configuration
create_sample_config() {
    if [ ! -f "config.yaml" ]; then
        print_status "Creating sample configuration..."
        cp "${CONFIG_FILE}" "config.yaml"
        
        # Update data directory in config
        sed -i.bak "s|data_dir: \"./data\"|data_dir: \"${DATA_DIR}\"|g" config.yaml
        rm config.yaml.bak 2>/dev/null || true
        
        print_success "Sample configuration created: config.yaml"
        print_warning "Please edit config.yaml with your specific settings"
    else
        print_warning "Configuration file already exists: config.yaml"
    fi
}

# Function to run tests (for dev installation)
run_tests() {
    if [ "$INSTALL_TYPE" = "dev" ] || [ "$INSTALL_TYPE" = "full" ]; then
        print_status "Running test suite..."
        
        # Run quick tests only
        ${PYTHON_CMD} -m pytest tests/test_quantitative_components.py -v --tb=short
        
        if [ $? -eq 0 ]; then
            print_success "Quick tests passed"
        else
            print_warning "Some tests failed - this is normal for initial setup"
        fi
    fi
}

# Function to print post-installation instructions
print_instructions() {
    echo ""
    echo "========================================"
    echo "Installation Complete!"
    echo "========================================"
    echo ""
    print_success "Factor Forecasting System has been installed successfully"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Activate the virtual environment:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. Prepare your data:"
    echo "   - Place Parquet files in: ${DATA_DIR}"
    echo "   - Ensure columns: date, stock_id, features, targets"
    echo ""
    echo "3. Configure the system:"
    echo "   - Edit: config.yaml"
    echo "   - Set your data paths and parameters"
    echo ""
    echo "4. Start training:"
    echo "   factor-train --config config.yaml"
    echo ""
    echo "5. Start API server (if installed):"
    if [ "$INSTALL_TYPE" = "api" ] || [ "$INSTALL_TYPE" = "full" ]; then
        echo "   factor-api --port 8000"
    else
        echo "   pip install -e .[api]  # Install API dependencies first"
        echo "   factor-api --port 8000"
    fi
    echo ""
    echo "For more information, see README.md"
    echo ""
}

# Main installation flow
main() {
    echo "Installation type: ${INSTALL_TYPE}"
    echo "Data directory: ${DATA_DIR}"
    echo "Config template: ${CONFIG_FILE}"
    echo ""
    
    check_python_version
    create_venv
    install_dependencies
    setup_directories
    validate_installation
    create_sample_config
    run_tests
    print_instructions
}

# Handle script arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Factor Forecasting System Installation Script"
    echo ""
    echo "Usage: $0 [INSTALL_TYPE] [DATA_DIR] [CONFIG_FILE]"
    echo ""
    echo "Arguments:"
    echo "  INSTALL_TYPE   Installation type (default: standard)"
    echo "                 Options: standard, gpu, dev, api, full"
    echo "  DATA_DIR       Data directory path (default: ./data)"
    echo "  CONFIG_FILE    Config template file (default: configs/training_configs/local/local_training.yaml)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Standard installation"
    echo "  $0 gpu                       # GPU-enabled installation"
    echo "  $0 dev                       # Development installation with test tools"
    echo "  $0 api                       # API server installation"
    echo "  $0 full /path/to/data        # Full installation with custom data path"
    echo ""
    exit 0
fi

# Run main installation
main

print_success "Installation script completed successfully!"
