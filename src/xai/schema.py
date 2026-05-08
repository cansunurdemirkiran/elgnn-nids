"""
xai/schema.py

Data structures for XAI operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SampleSelection:
    """
    SHAP sample selection data structure.
    """
    dataset_name: str
    background_samples: list[dict[str, Any]]
    explained_samples: list[dict[str, Any]]
    selection_metadata: dict[str, Any]
