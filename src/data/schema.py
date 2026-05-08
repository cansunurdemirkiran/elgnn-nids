"""
data/schema.py

Veri yükleme katmanının tip güvenli dönüş yapıları.

TrainSplit:  training seti (binary label)
EvalSplit:   validation / test seti (binary + multi-class label)
DatasetCollection: tüm split'lerin sözlüğü

Eski tuple tabanlı API'nin sorunları:
  - load_dataset() split'e göre farklı uzunlukta tuple döndürüyordu → tip hatası riski
  - Pozisyonel erişim (data[2]) kırılgandı
  - Yeni alan eklemek tüm çağrı noktalarını etkilerdi

Bu dataclass'lar her iki sorunu da çözer: named fields + type safety.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TrainSplit:
    """
    Training split verisi.

    Attributes:
        X:             (N, F) float32 feature tensörü.
        y:             (N,) long binary label tensörü (0=benign, 1=attack).
        feature_names: F uzunluklu feature adları listesi.
    """
    X: torch.Tensor
    y: torch.Tensor
    feature_names: list[str]


@dataclass(frozen=True)
class EvalSplit:
    """
    Validation veya test split verisi.

    Binary label modelin eğitim/değerlendirmesi için,
    multi-class label SHAP per-attack-type analizi için kullanılır.

    Attributes:
        X:             (N, F) float32 feature tensörü.
        y_binary:      (N,) long binary label tensörü (0=benign, 1=attack).
        feature_names: F uzunluklu feature adları listesi.
        y_multi:       (N,) orijinal multi-class label array'i (string veya int).
    """
    X: torch.Tensor
    y_binary: torch.Tensor
    feature_names: list[str]
    y_multi: np.ndarray


# Kolaylık takma adları
DatasetSplits = dict[str, "TrainSplit | EvalSplit"]
DatasetCollection = dict[str, DatasetSplits]
