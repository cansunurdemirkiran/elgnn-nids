"""
config package for ELGNN-NIDS project.

This package contains configuration files:
- features: Feature selection and drop lists
- hyperparams: Model hyperparameters and training settings
- paths: File system paths and constants
"""

from . import features
from . import hyperparams
from . import paths

__all__ = [
    "features",
    "hyperparams", 
    "paths",
]
