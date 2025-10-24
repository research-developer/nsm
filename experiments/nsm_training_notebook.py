"""
NSM Training Notebook - Interactive Modal Jupyter Environment

Launch with: modal run experiments/nsm_training_notebook.py
"""

import modal
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

app = modal.App("nsm-notebook")

# Build image with Jupyter and all NSM dependencies
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
        add_python="3.10"
    )
    .run_commands(
        "pip install --upgrade pip",
        # PyG dependencies for torch 2.1.0 + cu118
        "pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html",
    )
    .pip_install(
        "torch-geometric==2.4.0",
        "numpy",
        "scipy",
        "networkx",
        "matplotlib",
        "seaborn",
        "pandas",
        "tensorboard",
        "jupyter",
        "jupyterlab",
        "ipywidgets",
        "tqdm",
        "plotly",
        "kaleido",  # For plotly static image export
    )
    # Add NSM codebase
    .add_local_dir(PROJECT_ROOT / "nsm", remote_path="/root/nsm")
    # Add notebook
    .add_local_file(
        PROJECT_ROOT / "experiments" / "NSM_Training_Dashboard.ipynb",
        remote_path="/root/NSM_Training_Dashboard.ipynb"
    )
    .run_commands(
        # Enable widgets extension
        "jupyter nbextension enable --py widgetsnbextension --sys-prefix",
        # JupyterLab extensions
        "jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build || true",
    )
)

# Persistent volume for checkpoints and results
volume = modal.Volume.from_name("nsm-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/checkpoints": volume},
    timeout=14400,  # 4 hours
    cpu=4,
    memory=16_000,  # 16GB RAM for data loading
)
def notebook():
    """Launch Jupyter Lab with GPU access"""
    import subprocess
    import os

    # Set environment for optimal performance
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    print("=" * 60)
    print("🚀 NSM Training Notebook Starting")
    print("=" * 60)
    print("\n📊 Environment Info:")

    # Check GPU
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"  ✓ CUDA: {torch.version.cuda}")
    else:
        print("  ✗ No GPU detected!")

    print(f"\n📁 Volumes:")
    print(f"  ✓ Checkpoints: /checkpoints")

    # List existing checkpoints
    from pathlib import Path
    checkpoints = list(Path("/checkpoints").glob("**/*.pt"))
    print(f"  ✓ Found {len(checkpoints)} existing checkpoints")

    print("\n" + "=" * 60)
    print("🔗 Access your notebook via the URL below")
    print("=" * 60 + "\n")

    # Launch JupyterLab (better UI than classic notebook)
    subprocess.run([
        "jupyter", "lab",
        "--ip=0.0.0.0",
        "--port=8888",
        "--no-browser",
        "--allow-root",
        "--NotebookApp.token=''",  # No password for convenience
        "--NotebookApp.password=''",
        "--notebook-dir=/root",
    ])

@app.local_entrypoint()
def main():
    """Entry point for modal run"""
    print("\n🎯 Launching NSM Training Notebook...")
    print("⏳ This may take 1-2 minutes to provision GPU and load environment\n")
    notebook.remote()
