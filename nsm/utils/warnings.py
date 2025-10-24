"""
Warning suppression utilities for NSM.

Helps reduce noise from known non-critical warnings that clutter logs.
"""

import warnings
import os
import sys


def suppress_pyg_warnings():
    """
    Suppress PyTorch Geometric torch-scatter/torch-sparse import warnings.

    These warnings are non-critical - PyG has pure PyTorch fallbacks that
    work correctly. From NSM-31 analysis, SAGPooling works despite warnings.

    The warnings being suppressed:
    - "An issue occurred while importing 'torch-scatter'"
    - "An issue occurred while importing 'torch-sparse'"
    - Symbol not found errors from dlopen

    These occur on macOS ARM64 when compiled extensions don't match PyTorch
    version, but PyG gracefully falls back to pure PyTorch implementations.
    """
    # Suppress Python warnings from PyG
    warnings.filterwarnings('ignore', message='.*torch-scatter.*')
    warnings.filterwarnings('ignore', message='.*torch-sparse.*')
    warnings.filterwarnings('ignore', message='.*torch-cluster.*')
    warnings.filterwarnings('ignore', message='.*torch-spline-conv.*')

    # Suppress warnings about runpy module execution
    warnings.filterwarnings('ignore', category=RuntimeWarning,
                          message='.*found in sys.modules.*')


def suppress_all_nsm_warnings(verbose: bool = False):
    """
    Suppress all known non-critical NSM warnings.

    Args:
        verbose: If True, print what warnings are being suppressed

    Suppresses:
    - PyTorch Geometric extension import warnings
    - UserWarnings from torch_geometric about missing extensions
    - RuntimeWarnings about module imports
    """
    if verbose:
        print("Suppressing non-critical warnings:")
        print("  - PyTorch Geometric extension imports")
        print("  - Module import runtime warnings")

    suppress_pyg_warnings()


def configure_warnings(
    suppress_pyg: bool = True,
    suppress_all: bool = False,
    verbose: bool = False
):
    """
    Configure warning behavior for NSM training/evaluation.

    Args:
        suppress_pyg: Suppress PyG extension warnings (default True)
        suppress_all: Suppress all non-critical warnings (default False)
        verbose: Print configuration info (default False)

    Example:
        >>> from nsm.utils.warnings import configure_warnings
        >>> configure_warnings(suppress_pyg=True)
    """
    if suppress_all:
        suppress_all_nsm_warnings(verbose=verbose)
    elif suppress_pyg:
        suppress_pyg_warnings()
        if verbose:
            print("Suppressing PyG extension warnings")


# Auto-suppress on import if NSM_SUPPRESS_WARNINGS env var is set
if os.getenv('NSM_SUPPRESS_WARNINGS', '1') == '1':
    suppress_all_nsm_warnings(verbose=False)
