"""
preprocessing/transforms.py

Sayısal dönüşüm işlevleri — log1p ve MinMaxScaler sarmalayıcıları.
Tüm dönüşümler in-place veya yeni dizi döndürerek gerçekleşir.

Public API:
    log1p_safe(X)                    → np.ndarray  (in-place, yeni referans)
    fit_and_scale(X_train, X_test)   → (np.ndarray, np.ndarray, MinMaxScaler)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from ..config import features


def log1p_safe(X: np.ndarray) -> np.ndarray:
    """
    Negatif değerleri 0'a clamp edip log1p uygular (in-place).

    Sıralama önemli: önce clip, sonra log1p — negatif değer için
    log(1 + x < 0) = NaN üretmez.

    Args:
        X: (N, F) float32 feature matrisi.

    Returns:
        Dönüştürülmüş X (aynı bellek alanı, yeni referans).
    """
    np.maximum(X, 0.0, out=X)
    np.log1p(X, out=X)
    return X


def fit_and_scale(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    MinMaxScaler'ı sadece train üzerine fit et, her iki seti transform et.

    Train-test leakage önleme prensibi: scaler yalnızca train verisi
    üzerinde fit edilir, test verisi sadece transform edilir.

    Args:
        X_train:       (N_train, F) float32 array.
        X_test:        (N_test, F) float32 array.
        feature_range: Ölçekleme aralığı, varsayılan (0, 1).

    Returns:
        (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    return X_train_scaled, X_test_scaled, scaler


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

    for category, cols in features.get_drop_categories().items():
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
