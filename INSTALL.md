# NSM Installation Guide

This guide covers setting up the NSM development environment using conda, which can be shared across all git worktrees.

## Quick Start

```bash
# 1. Clone the repository
git clone <repository-url> nsm
cd nsm

# 2. Run the setup script (recommended)
./setup.sh

# 3. Activate the environment
conda activate nsm

# 4. Verify installation
python -c "import torch; import torch_geometric; print('Success!')"
```

## Prerequisites

### Required

- **Conda**: Miniconda or Anaconda
  - Install from: https://docs.conda.io/en/latest/miniconda.html
- **CUDA 11.8** (for GPU support)
  - Check version: `nvidia-smi`
  - If using different CUDA version, modify URLs in `requirements.txt` and `environment.yml`

### Optional

- **Graphviz** (for graph visualization)
  - macOS: `brew install graphviz`
  - Ubuntu/Debian: `sudo apt-get install graphviz`
  - Windows: Download from https://graphviz.org/download/

## Installation Methods

### Method 1: Using setup.sh (Recommended)

The setup script automates environment creation:

```bash
./setup.sh              # Create new environment
./setup.sh --update     # Update existing environment
./setup.sh --clean      # Remove and recreate environment
```

### Method 2: Manual conda installation

```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate nsm
```

### Method 3: Using pip only (not recommended)

If you cannot use conda:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch first (adjust CUDA version as needed)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

## Verification

After installation, verify everything works:

```bash
conda activate nsm

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check PyTorch Geometric
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run basic tests
pytest tests/ -v  # (once tests are created)
```

Expected output:
```
PyTorch: 2.1.0
PyG: 2.4.0
CUDA available: True  # Or False if no GPU
```

## Using Across Worktrees

The conda environment is **shared across all git worktrees**. You only need to install it once:

```bash
# Main repository
cd /Users/preston/Projects/NSM
./setup.sh
conda activate nsm

# In any worktree
cd ../nsm-planning  # or any other worktree
conda activate nsm  # Same environment!
```

This ensures:
- Consistent dependencies across all branches
- No need to reinstall packages per worktree
- Faster worktree creation
- Reduced disk space usage

## Environment Management

### Updating the environment

When dependencies change:

```bash
# Pull latest changes
git pull

# Update environment
./setup.sh --update

# Or manually
conda env update -f environment.yml --prune
```

### Exporting your environment

To share exact package versions:

```bash
# Export full environment (platform-specific)
conda env export > environment-lock.yml

# Export cross-platform
conda env export --from-history > environment-minimal.yml
```

### Removing the environment

```bash
conda deactivate
conda env remove -n nsm
```

## Troubleshooting

### CUDA version mismatch

If you have a different CUDA version:

1. Check your CUDA version: `nvidia-smi`
2. Update PyTorch Geometric URLs in `environment.yml` and `requirements.txt`
3. Find correct versions at: https://data.pyg.org/whl/

Example for CUDA 12.1:
```yaml
- torch-scatter==2.1.2+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### Conda is slow

Use libmamba solver (faster):

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

### Import errors with PyG extensions

If you get import errors for `torch_scatter`, `torch_sparse`, etc.:

```bash
# Reinstall with correct CUDA version
pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

### macOS ARM (M1/M2) notes

PyTorch Geometric may have limited support on Apple Silicon. Consider:

1. Use CPU-only version (slower):
   ```bash
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
       -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
   ```

2. Or use x86_64 conda (via Rosetta):
   ```bash
   CONDA_SUBDIR=osx-64 conda create -n nsm python=3.10
   conda activate nsm
   conda config --env --set subdir osx-64
   ```

### Out of memory during installation

If conda runs out of memory:

```bash
# Clean conda cache
conda clean --all

# Install in stages
conda create -n nsm python=3.10
conda activate nsm
conda install pytorch=2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Development Setup

For active development, also install:

```bash
# Pre-commit hooks (code quality)
pre-commit install

# Jupyter kernel
python -m ipykernel install --user --name nsm --display-name "NSM"
```

## GPU Configuration

### Check GPU status

```bash
# NVIDIA GPU info
nvidia-smi

# PyTorch GPU info
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

### Multi-GPU setup

For multi-GPU training:

```bash
# Check available GPUs
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Set specific GPU
export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0
```

## Next Steps

After successful installation:

1. Read `CLAUDE.md` for architecture overview
2. Check `README.md` for project background
3. Review implementation issues in Linear (NSM-20 and sub-issues)
4. Start with NSM-18 (PyG Environment & Data Structures)

## Support

For installation issues:
- Check Linear issues: https://linear.app/imajn/team/NSM/all
- PyTorch docs: https://pytorch.org/get-started/locally/
- PyG docs: https://pytorch-geometric.readthedocs.io/
