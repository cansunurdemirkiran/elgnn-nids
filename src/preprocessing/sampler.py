"""
preprocessing/sampler.py

Sınıf dengesizliği giderme — SMOTE oversampling + ENN cleanup.
Sadece training setine uygulanır; test/val setlerine dokunulmaz.

Public API:
    run_smote_enn(X_train, y_train) → (np.ndarray, np.ndarray)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours


def run_smote_enn(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Training setine sırayla SMOTE sonra ENN uygular.

    SMOTE: azınlık sınıfı sentetik örneklerle çoğaltır.
    ENN:   sınır yakınındaki gürültülü örnekleri temizler.

    Uygulama sırası önemlidir: önce over-sample, sonra noise cleanup.

    Args:
        X_train: (N, F) float32 training feature matrisi.
        y_train: (N,) int binary training label'ı.

    Returns:
        (X_resampled, y_resampled) — dengelenmiş training seti.
    """
    print(
        f"      before SMOTE: {pd.Series(y_train).value_counts().to_dict()}",
        flush=True,
    )
    smote = SMOTE()
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    print(
        f"      after SMOTE:  {pd.Series(y_smote).value_counts().to_dict()}",
        flush=True,
    )

    print(
        f"      before ENN: {pd.Series(y_smote).value_counts().to_dict()}"
        f"  shape={X_smote.shape}",
        flush=True,
    )
    enn = EditedNearestNeighbours()
    X_resampled, y_resampled = enn.fit_resample(X_smote, y_smote)
    print(
        f"      after ENN:  {pd.Series(y_resampled).value_counts().to_dict()}"
        f"  shape={X_resampled.shape}",
        flush=True,
    )

    return X_resampled, y_resampled
