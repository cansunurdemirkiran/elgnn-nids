"""
data package for ELGNN-NIDS project.

This package contains data loading and schema definitions:
- loader: CSV loading and dataset management
- schema: Data structures for splits and collections
- transforms: Data transformation utilities
"""

from .loader import load_all_datasets, load_dataset, sanity_check
from .schema import DatasetCollection, EvalSplit, TrainSplit

__all__ = [
    "load_dataset",
    "load_all_datasets", 
    "sanity_check",
    "DatasetCollection",
    "EvalSplit",
    "TrainSplit",
]
