"""
ELGNN-NIDS: Graph Neural Networks for Network Intrusion Detection

This project compares three different GNN architectures for IDS:
- GCN (baseline)
- GCN + EWC (continual learning)  
- E-GraphSAGE (edge-feature supported, continual learning)

The comparison is made on two axes: classification performance and SHAP feature importance.

All models inherit from BaseGNN → same training loop, same SHAP pipeline.
"""

# Re-export main components from src package
from src.models import (
    BaseGNN,
    ELGNN,
    EGraphSAGE,
    count_parameters,
    model_summary,
    count_parameters_sage,
    model_summary_sage,
)

# Re-export data loading components
from src.data import (
    load_dataset,
    load_all_datasets,
    sanity_check,
    DatasetCollection,
    EvalSplit,
    TrainSplit,
)

# Re-export preprocessing components
from src.preprocessing import (
    pipeline_main,
    clean,
    encode_categoricals_keep_label,
    load_raw_csv,
    write_csv,
    run_smote_enn,
    build_split_datasets,
    fit_and_scale,
    log1p_safe,
)

# Re-export XAI components
from src.xai import (
    selector_main,
    SampleSelection,
)

# Re-export utilities
from src.utils import (
    banner,
    step,
)

# Re-export config
from src.config import (
    features,
    hyperparams,
    paths,
)

__all__ = [
    # Models
    "BaseGNN",
    "ELGNN", 
    "EGraphSAGE",
    "count_parameters",
    "model_summary",
    "count_parameters_sage",
    "model_summary_sage",
    # Data
    "load_dataset",
    "load_all_datasets",
    "sanity_check",
    "DatasetCollection",
    "EvalSplit", 
    "TrainSplit",
    # Preprocessing
    "pipeline_main",
    "clean",
    "encode_categoricals_keep_label",
    "load_raw_csv",
    "write_csv",
    "run_smote_enn",
    "build_split_datasets",
    "fit_and_scale",
    "log1p_safe",
    # XAI
    "selector_main",
    "SampleSelection",
    # Utils
    "banner",
    "step",
    # Config
    "features",
    "hyperparams",
    "paths",
]
