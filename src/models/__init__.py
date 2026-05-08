"""
models package for ELGNN-NIDS project.

This package contains the Graph Neural Network implementations:
- BaseGNN: Abstract base class for all GNN models
- ELGNN: GCN-based model (baseline and EWC variants)
- EGraphSAGE: Edge-feature enhanced GraphSAGE model
"""

from .base import BaseGNN
from .e_graphsage import EGraphSAGE, count_parameters as count_parameters_sage, model_summary as model_summary_sage
from .elgnn import ELGNN, count_parameters, model_summary

__all__ = [
    # Abstract interface
    "BaseGNN",
    # Model classes
    "ELGNN",
    "EGraphSAGE",
    # Utilities
    "count_parameters",
    "model_summary",
    "count_parameters_sage",
    "model_summary_sage",
]
