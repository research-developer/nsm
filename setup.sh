#!/bin/bash
# NSM Environment Setup Script
#
# This script creates a conda environment that can be shared across
# all git worktrees for the NSM project.
#
# Usage:
#   ./setup.sh              # Create environment
#   ./setup.sh --update     # Update existing environment
#   ./setup.sh --clean      # Remove and recreate environment

set -e  # Exit on error

ENV_NAME="nsm"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[NSM]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[NSM]${NC} $1"
}

print_error() {
    echo -e "${RED}[NSM]${NC} $1"
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "Conda not found. Please install Miniconda or Anaconda first."
    print_status "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Parse arguments
CLEAN=false
UPDATE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --update)
            UPDATE=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Usage: $0 [--clean|--update]"
            exit 1
            ;;
    esac
done

# Clean environment if requested
if [ "$CLEAN" = true ]; then
    print_warning "Removing existing environment: $ENV_NAME"
    conda env remove -n $ENV_NAME -y 2>/dev/null || true
    UPDATE=false
fi

# Check if environment exists
if conda env list | grep -q "^$ENV_NAME "; then
    if [ "$UPDATE" = true ]; then
        print_status "Updating environment: $ENV_NAME"
        conda env update -f "$SCRIPT_DIR/environment.yml" --prune
    else
        print_warning "Environment '$ENV_NAME' already exists."
        print_status "Use --update to update it or --clean to recreate it."
        exit 0
    fi
else
    print_status "Creating conda environment: $ENV_NAME"
    conda env create -f "$SCRIPT_DIR/environment.yml"
fi

print_status "Environment setup complete!"
echo ""
print_status "To activate the environment, run:"
echo "    conda activate $ENV_NAME"
echo ""
print_status "To verify installation:"
echo "    conda activate $ENV_NAME"
echo "    python -c 'import torch; import torch_geometric; print(f\"PyTorch: {torch.__version__}\"); print(f\"PyG: {torch_geometric.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'"
echo ""

# Check for system dependencies
print_status "Checking system dependencies..."

if ! command -v dot &> /dev/null; then
    print_warning "Graphviz not found (optional, for graph visualization)"
    print_status "Install with: brew install graphviz (macOS) or apt-get install graphviz (Linux)"
fi

# Verify CUDA if available
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    print_warning "No NVIDIA GPU detected. Training will use CPU (slower)."
fi

print_status "Setup complete! The environment can be used across all worktrees."
