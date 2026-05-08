"""
data/transforms.py

Yüklü DataFrame üzerindeki dönüşümler:
  - Feature drop (config.features'dan)
  - Binary label encoding
  - PyTorch tensor dönüşümü

Bu modül preprocessing'den bağımsızdır: preprocessed CSV'ler
üzerinde çalışır, ham veriyle ilgilenmez.

Public API:
    apply_drops(df, verbose)        → pd.DataFrame
    encode_binary_labels(y)         → np.ndarray
    to_tensors(X, y)                → (torch.Tensor, torch.Tensor)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import config


def apply_drops(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    config.features'daki tüm drop kategorilerini DataFrame'e uygula.

    Var olmayan sütunlar sessizce atlanır; bu sayede farklı veri setlerinde
    eksik sütunlar hata vermez.

    Args:
        df:      Sütunları düşürülecek DataFrame.
        verbose: Her kategori için kaç sütun düşürüldüğünü yazdırır.

    Returns:
        Belirtilen sütunlar düşürülmüş yeni DataFrame.
    """
    df = df.copy()
    total_dropped = 0

    for category, cols in config.get_drop_categories().items():
        existing = [c for c in cols if c in df.columns]
        if existing:
            df = df.drop(columns=existing)
            total_dropped += len(existing)
            if verbose:
                print(f"  [{category:18s}] dropped {len(existing):2d}")

    if verbose:
        print(f"  Total dropped: {total_dropped}")

    return df


def encode_binary_labels(y: np.ndarray) -> np.ndarray:
    """
    Multi-class veya binary label'ı {0, 1} integer array'ine çevir.

    Karar mantığı:
      - Zaten sayısal ve {0, 1} kümesine dahilse → doğrudan int64'e çevir.
      - Diğer sayısal değerlerse → sıfır dışı = 1.
      - String ise → "benign" / "normal" = 0, diğer her şey = 1.

    Args:
        y: Ham label array'i (herhangi bir dtype).

    Returns:
        int64 dtype, {0, 1} değerli binary label array'i.
    """
    if y.dtype.kind in ("i", "u", "f"):
        unique_vals = set(np.unique(y).tolist())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            return y.astype(np.int64)
        return (y != 0).astype(np.int64)

    # String label
    benign_keywords = {"benign", "normal"}
    binary = np.array([
        0 if str(label).strip().lower() in benign_keywords else 1
        for label in y
    ])
    return binary.astype(np.int64)


def to_tensors(
    X: pd.DataFrame,
    y: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Feature DataFrame ve label array'ini PyTorch tensörlerine çevir.

    Args:
        X: (N, F) feature DataFrame.
        y: (N,) int label array'i.

    Returns:
        (X_tensor: float32, y_tensor: long)
    """
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor
