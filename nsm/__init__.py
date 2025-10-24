"""
Neural Symbolic Model (NSM)

A neurosymbolic language model architecture using recursive semantic
triple decomposition across a mathematically-grounded hierarchy.
"""

__version__ = "0.1.0"

# Suppress non-critical warnings by default (can be disabled with NSM_SUPPRESS_WARNINGS=0)
from nsm.utils.warnings import configure_warnings
configure_warnings(suppress_pyg=True)

from . import data

__all__ = ["data"]
