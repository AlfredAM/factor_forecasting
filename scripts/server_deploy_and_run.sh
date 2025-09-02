#!/bin/bash

# Server connection and deployment script
set -e

# Server configuration
SERVER_IP="8.155.163.64"
SERVER_USER="ecs-user"
SERVER_PASSWORD="Abab1234"
PROJECT_NAME="factor_forecasting"
LOCAL_PROJECT_DIR="/Users/scratch/Documents/My Code/Projects/factor_forecasting"
REMOTE_BASE_DIR="/nas"
REMOTE_PROJECT_DIR="$REMOTE_BASE_DIR/$PROJECT_NAME"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "\n${BLUE}===========================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}===========================================${NC}\n"
}

# Check network connection
check_connection() {
    log_section "Checking server connection"
    
    log_info "Pinging server $SERVER_IP..."
    if ping -c 3 -W 5000 $SERVER_IP > /dev/null 2>&1; then
        log_info " Server network connected"
    else
        log_warn "Server ping failed, trying SSH connection..."
    fi
    
    log_info "Testing SSH connection..."
    if sshpass -p "$SERVER_PASSWORD" ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$SERVER_USER@$SERVER_IP" "echo 'SSH connection successful'" > /dev/null 2>&1; then
        log_info " SSH connection successful"
        return 0
    else
        log_error " SSH connection failed"
        return 1
    fi
}

# Wait for server connection
wait_for_connection() {
    log_section "Waiting for server connection"
    
    local max_attempts=60
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log_info "Attempting connection ($attempt/$max_attempts)..."
        
        if check_connection; then
            log_info " Server connection successful!"
            return 0
        fi
        
        log_warn "Connection failed, retrying in 30 seconds..."
        sleep 30
        ((attempt++))
    done
    
    log_error "Reached maximum retry attempts, unable to connect to server"
    exit 1
}

# Check or install sshpass
check_sshpass() {
    if ! command -v sshpass &> /dev/null; then
        log_warn "sshpass not installed, installing..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command -v brew &> /dev/null; then
                brew install hudochenkov/sshpass/sshpass
            else
                log_error "Please install Homebrew first, then run: brew install hudochenkov/sshpass/sshpass"
                exit 1
            fi
        else
            # Linux
            sudo apt-get update && sudo apt-get install -y sshpass
        fi
    fi
}

# Clear related processes on server
kill_server_processes() {
    log_section "Clear related processes on server"
    
    log_info "Finding and terminating Python training processes..."
    sshpass -p "$SERVER_PASSWORD" ssh -o StrictHostKeyChecking=no "$SERVER_USER@$SERVER_IP" "
        # Find related processes
        echo 'Finding related Python processes:'
        ps aux | grep -E 'python.*factor_forecasting|python.*tcn|python.*train' | grep -v grep || echo 'No related processes found'
        
        # Terminate processes
        pkill -f 'python.*factor_forecasting' || echo 'No factor_forecasting process found'
        pkill -f 'python.*tcn' || echo 'No tcn related process found'
        pkill -f 'python.*train' || echo 'No train related process found'
        pkill -f 'server_optimized_train' || echo 'No optimized training process found'
        
        # Clean up GPU processes
        if command -v nvidia-smi &> /dev/null; then
            echo 'Checking GPU processes:'
            nvidia-smi pids || echo 'nvidia-smi not available or no GPU processes'
        fi
        
        echo 'Process cleanup complete'
    "
    
    log_info " Server process cleanup complete"
}

# Check server environment
check_server_environment() {
    log_section "Check server environment"
    
    log_info "Getting server system information..."
    sshpass -p "$SERVER_PASSWORD" ssh -o StrictHostKeyChecking=no "$SERVER_USER@$SERVER_IP" "
        echo '=== System Information ==='
        uname -a
        echo
        
        echo '=== CPU Information ==='
        nproc
        cat /proc/cpuinfo | grep 'model name' | head -1
        echo
        
        echo '=== Memory Information ==='
        free -h
        echo
        
        echo '=== Disk Space ==='
        df -h /nas 2>/dev/null || df -h /
        echo
        
        echo '=== GPU Information ==='
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
        else
            echo 'NVIDIA GPU not available'
        fi
        echo
        
        echo '=== Python Environment ==='
        python3 --version 2>/dev/null || echo 'Python3 not available'
        pip3 --version 2>/dev/null || echo 'pip3 not available'
        echo
        
        echo '=== Conda Environment ==='
        if [ -d /nas/miniconda3 ]; then
            echo 'Found Conda installation: /nas/miniconda3'
            ls -la /nas/miniconda3/bin/conda 2>/dev/null || echo 'conda command not found'
        elif [ -d /nas/anaconda3 ]; then
            echo 'Found Anaconda installation: /nas/anaconda3'
            ls -la /nas/anaconda3/bin/conda 2>/dev/null || echo 'conda command not found'
        else
            echo 'No Conda installation found under /nas'
            find /nas -name conda -type f 2>/dev/null | head -5 || echo 'No conda found in /nas'
        fi
    "
}

# Activate Conda environment
activate_conda() {
    log_section "Activate Conda environment"
    
    log_info "Finding and activating Conda..."
    sshpass -p "$SERVER_PASSWORD" ssh -o StrictHostKeyChecking=no "$SERVER_USER@$SERVER_IP" "
        # Find conda installation path
        CONDA_PATH=''
        if [ -f /nas/miniconda3/bin/conda ]; then
            CONDA_PATH='/nas/miniconda3'
        elif [ -f /nas/anaconda3/bin/conda ]; then
            CONDA_PATH='/nas/anaconda3'
        else
            # Try to find other locations
            CONDA_SEARCH=\$(find /nas -name 'conda' -type f 2>/dev/null | head -1)
            if [ -n \"\$CONDA_SEARCH\" ]; then
                CONDA_PATH=\$(dirname \$(dirname \$CONDA_SEARCH))
            fi
        fi
        
        if [ -z \"\$CONDA_PATH\" ]; then
            echo 'ERROR: Conda installation not found'
            exit 1
        fi
        
        echo \"Found Conda installation: \$CONDA_PATH\"
        
        # Initialize conda
        source \$CONDA_PATH/etc/profile.d/conda.sh
        
        # Check environment
        conda info --envs || echo 'Unable to get conda environment list'
        
        # Create or activate project environment
        ENV_NAME='factor_forecasting'
        if conda env list | grep -q \$ENV_NAME; then
            echo \"Activating existing environment: \$ENV_NAME\"
            conda activate \$ENV_NAME
        else
            echo \"Creating new environment: \$ENV_NAME\"
            conda create -n \$ENV_NAME python=3.9 -y
            conda activate \$ENV_NAME
        fi
        
        # Verify environment
        echo \"Current Python path: \$(which python)\"
        echo \"Current Python version: \$(python --version)\"
        
        # Save conda path to file for future use
        echo \$CONDA_PATH > /nas/conda_path.txt
        echo \$ENV_NAME > /nas/conda_env.txt
    "
    
    log_info " Conda environment activation complete"
}

# Deploy project
deploy_project() {
    log_section "Deploy project to server"
    
    log_info "Preparing project files..."
    cd "$LOCAL_PROJECT_DIR"
    
    # Create temporary packaging directory
    TEMP_DIR="/tmp/${PROJECT_NAME}_deploy_$(date +%s)"
    mkdir -p "$TEMP_DIR"
    
    # Copy project files
    rsync -av --exclude='venv/' \
              --exclude='__pycache__/' \
              --exclude='*.pyc' \
              --exclude='.git/' \
              --exclude='outputs/' \
              --exclude='*.log' \
              --exclude='.DS_Store' \
              --exclude='*.egg-info/' \
              --exclude='dist/' \
              . "$TEMP_DIR/"
    
    # Create compressed package
    cd /tmp
    tar -czf "${PROJECT_NAME}_deploy.tar.gz" "$(basename $TEMP_DIR)/"
    
    log_info "Uploading project files to server..."
    sshpass -p "$SERVER_PASSWORD" scp -o StrictHostKeyChecking=no "/tmp/${PROJECT_NAME}_deploy.tar.gz" "$SERVER_USER@$SERVER_IP:/nas/"
    
    log_info "Unpacking and setting up project on server..."
    sshpass -p "$SERVER_PASSWORD" ssh -o StrictHostKeyChecking=no "$SERVER_USER@$SERVER_IP" "
        cd /nas
        
        # Backup old project
        if [ -d $PROJECT_NAME ]; then
            mv $PROJECT_NAME ${PROJECT_NAME}_backup_$(date +%Y%m%d_%H%M%S) || echo 'Backup failed, continuing...'
        fi
        
        # Unpack new project
        tar -xzf ${PROJECT_NAME}_deploy.tar.gz
        mv ${PROJECT_NAME}_deploy_* $PROJECT_NAME
        cd $PROJECT_NAME
        
        echo 'Project files deployed, current directory:'
        pwd
        ls -la
        
        # Activate conda environment
        CONDA_PATH=\$(cat /nas/conda_path.txt 2>/dev/null || echo '/nas/miniconda3')
        ENV_NAME=\$(cat /nas/conda_env.txt 2>/dev/null || echo 'factor_forecasting')
        
        source \$CONDA_PATH/etc/profile.d/conda.sh
        conda activate \$ENV_NAME
        
        # Install dependencies
        echo 'Installing Python dependencies...'
        pip install --upgrade pip
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install -r requirements.txt
        pip install -e .
        
        echo 'Dependencies installed'
        echo 'Current Python environment:'
        which python
        python --version
        pip list | grep -E 'torch|pandas|numpy' || echo 'Check related packages'
    "
    
    # Clean up local temporary files
    rm -rf "$TEMP_DIR"
    rm -f "/tmp/${PROJECT_NAME}_deploy.tar.gz"
    
    log_info " Project deployment complete"
}

# Set training parameters and start training
start_training() {
    log_section "Start model training"
    
    log_info "Starting background training..."
    sshpass -p "$SERVER_PASSWORD" ssh -o StrictHostKeyChecking=no "$SERVER_USER@$SERVER_IP" "
        cd /nas/$PROJECT_NAME
        
        # Activate conda environment
        CONDA_PATH=\$(cat /nas/conda_path.txt 2>/dev/null || echo '/nas/miniconda3')
        ENV_NAME=\$(cat /nas/conda_env.txt 2>/dev/null || echo 'factor_forecasting')
        
        source \$CONDA_PATH/etc/profile.d/conda.sh
        conda activate \$ENV_NAME
        
        # Check data directory
        DATA_PATH='/nas/feature_v2_10s'
        if [ ! -d \"\$DATA_PATH\" ]; then
            echo 'Data directory does not exist, please ensure NAS data path is correct...'
            echo 'Expected data path: /nas/feature_v2_10s'
            exit 1
        fi
        
        echo \"Using data path: \$DATA_PATH\"
        ls -la \$DATA_PATH
        
        # Create output directory
        OUTPUT_DIR='/nas/$PROJECT_NAME/outputs'
        mkdir -p \$OUTPUT_DIR
        
        # Create startup script
        cat > start_training.sh << 'EOF'
#!/bin/bash
source /nas/conda_path.txt > /dev/null 2>&1 || CONDA_PATH='/nas/miniconda3'
ENV_NAME=\$(cat /nas/conda_env.txt 2>/dev/null || echo 'factor_forecasting')

source \$CONDA_PATH/etc/profile.d/conda.sh
conda activate \$ENV_NAME

cd /nas/$PROJECT_NAME

python scripts/server_optimized_train.py \\
    --data-path \$DATA_PATH \\
    --output-dir \$OUTPUT_DIR \\
    --log-level INFO \\
    --device auto \\
    --epochs 50 \\
    > training_output.log 2>&1 &

echo \$! > training.pid
echo \"Training process started, PID: \$(cat training.pid)\"
echo \"Log file: \$(pwd)/training_output.log\"
echo \"Output directory: \$OUTPUT_DIR\"
EOF
        
        chmod +x start_training.sh
        
        # Start training
        echo 'Starting training process...'
        ./start_training.sh
        
        # Wait a bit to ensure process starts
        sleep 5
        
        # Check process status
        if [ -f training.pid ]; then
            PID=\$(cat training.pid)
            if ps -p \$PID > /dev/null; then
                echo \" Training process running (PID: \$PID)\"
            else
                echo \" Training process startup failed\"
                tail -n 20 training_output.log || echo 'Unable to read log'
            fi
        else
            echo \" No process ID file found\"
        fi
    "
    
    log_info " Training started"
}

# Monitor training logs
monitor_logs() {
    log_section "Monitor training logs"
    
    log_info "Starting log monitoring (press Ctrl+C to stop monitoring)..."
    
    while true; do
        log_info "=== $(date) ==="
        
        # Check process status
        sshpass -p "$SERVER_PASSWORD" ssh -o StrictHostKeyChecking=no "$SERVER_USER@$SERVER_IP" "
            cd /nas/$PROJECT_NAME
            
            if [ -f training.pid ]; then
                PID=\$(cat training.pid)
                if ps -p \$PID > /dev/null; then
                    echo \"[Process Status] Training process running (PID: \$PID)\"
                else
                    echo \"[Process Status] Training process stopped\"
                fi
            else
                echo \"[Process Status] No process ID file found\"
            fi
            
            # System resources
            echo \"  CPU: \$(top -bn1 | grep 'Cpu(s)' | awk '{print \$2}' | cut -d'%' -f1)% usage\"
            echo \"  Memory: \$(free | grep Mem | awk '{printf \"%.1f%%\", \$3/\$2 * 100.0}') usage\"
            
            # GPU status
            if command -v nvidia-smi &> /dev/null; then
                echo \"[GPU Status]\"
                nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | head -1 | awk -F',' '{printf \"  GPU usage: %s%%, Memory: %s/%s MB, Temperature: %s°C\n\", \$1, \$2, \$3, \$4}'
            fi
            
            # Training logs (last 20 lines)
            echo \"[Training Logs]\"
            if [ -f training_output.log ]; then
                tail -n 10 training_output.log
            else
                echo \"  Log file not found\"
            fi
            
            echo \"----------------------------------------\"
        "
        
        sleep 30
    done
}

# Main function
main() {
    log_section "Optimized server training deployment script"
    
    # Check dependencies
    check_sshpass
    
    # Wait for server connection
    wait_for_connection
    
    # Check server environment
    check_server_environment
    
    # Clear old processes
    kill_server_processes
    
    # Activate Conda environment
    activate_conda
    
    # Deploy project
    deploy_project
    
    # Start training
    start_training
    
    log_section "Deployment complete"
    log_info "Project successfully deployed to server and training started"
    log_info "Monitoring commands:"
    log_info "  View logs: tail -f /nas/$PROJECT_NAME/training_output.log"
    log_info "  Check processes: ps aux | grep python"
    log_info "  Stop training: kill $(cat /nas/$PROJECT_NAME/training.pid)"
    log_info ""
    
    # Ask if monitoring should start
    read -p "Start monitoring training logs? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        monitor_logs
    else
        log_info " Script complete. You can monitor later using the command:"
        log_info "  ./$(basename $0) --monitor"
    fi
}

# Monitor only mode
monitor_only() {
    log_section "Monitor only mode"
    
    if check_connection; then
        monitor_logs
    else
        log_error "Unable to connect to server"
        exit 1
    fi
}

# Parse command line arguments
case "${1:-}" in
    --monitor)
        monitor_only
        ;;
    --help|-h)
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  --monitor    Monitor training logs only"
        echo "  --help       Show help"
        echo ""
        echo "Default: Complete deployment and training process"
        ;;
    *)
        main
        ;;
esac
