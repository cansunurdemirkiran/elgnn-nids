"""
xai package for ELGNN-NIDS project.

This package contains explainable AI components:
- selector: SHAP sample selection utilities
- schema: Data structures for XAI operations
"""

from .selector import main as selector_main
from .schema import SampleSelection

__all__ = [
    "selector_main",
    "SampleSelection",
]
