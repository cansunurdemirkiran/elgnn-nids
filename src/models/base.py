"""
models/base.py

GNN modellerinin soyut temel sınıfı.

Projede karşılaştırılan üç model bu sınıftan türer:
  - GCN          (models/elgnn.py)     — spektral graph convolution, EWC yok
  - GCN + EWC    (models/elgnn.py)     — aynı mimari, eğitimde EWC penalty eklenir
  - E-GraphSAGE  (models/e_graphsage.py) — edge-feature destekli GraphSAGE

SOLID bağlamı:
  - train.py, ewc.py, xai/ → BaseGNN'e bağımlı, somut sınıfa değil (DIP)
  - Yeni model eklemek bu dosyayı değiştirmez (OCP)
  - Tüm modeller aynı forward/predict_proba sözleşmesine uyar (LSP)

edge_attr parametresi opsiyoneldir:
  - GCN / GCN+EWC → edge_attr'ı yok sayar
  - E-GraphSAGE   → edge_attr'ı aggregation'a dahil eder
  Bu sayede tek bir training loop tüm modelleri çalıştırabilir.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseGNN(ABC, nn.Module):
    """
    Graph Neural Network modelleri için soyut temel sınıf.

    Alt sınıflar implement etmek zorundadır:
      - forward(x, edge_index, edge_attr=None) → logits (N, num_classes)
      - predict_proba(x, edge_index, edge_attr=None) → softmax probs (N, num_classes)

    Attributes:
        variant:    Model varyant adı — checkpoint ve raporlama için.
        input_dim:  Girdi feature boyutu.
        num_classes: Çıktı sınıf sayısı.
    """

    variant: str = "base"

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:          (N, input_dim) — node feature matrisi
            edge_index: (2, E) — edge listesi (PyG COO formatı)
            edge_attr:  (E, edge_dim) — edge feature matrisi (opsiyonel)
                        GCN/EWC-GCN bunu yok sayar; E-GraphSAGE kullanır.

        Returns:
            logits: (N, num_classes)
        """
        ...

    @abstractmethod
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Softmax olasılıkları — SHAP / LIME açıklanabilirlik için.

        Args:
            x:          (N, input_dim)
            edge_index: (2, E)
            edge_attr:  (E, edge_dim) opsiyonel

        Returns:
            probs: (N, num_classes) — satır toplamları 1.0
        """
        ...
