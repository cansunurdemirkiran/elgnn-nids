"""
models/elgnn.py

EL-GNN Mimarisi (Variant 1: Pure GCN, attention-free).

EL-GNN paper'ının orijinal mimarisine sadık implementasyon:

  Input (N, F)
      ↓
  GCN Layer 1 (F → H)  + ReLU + Dropout
      ↓
  GCN Layer 2 (H → O)  + ReLU + Dropout
      ↓
  Linear (O → O/2) + ReLU
      ↓
  Linear (O/2 → num_classes)
      ↓
  (CrossEntropyLoss, eğitim dışında)

NOT — İlerideki varyantlar:
  Variant 2: GAT (multi-head, komşu-bazlı attention)
  Variant 3: GCN + feature-level attention (manuel)
  → Her biri models/gat.py, models/attention_gcn.py olarak eklenir;
    BaseGNN'i genişletir, bu dosyaya dokunmaz.

EWC penalty hesabı ewc.py'da, training train.py'da.

Public API:
    ELGNN(input_dim, ...)
    count_parameters(model)  → int
    model_summary(model)     → None  (stdout)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    raise ImportError(
        "torch-geometric is not installed. Install it with:\n"
        "  pip install torch-geometric"
    )

from src.config import hyperparams as config
from .base import BaseGNN


class ELGNN(BaseGNN):
    """
    EL-GNN — Pure GCN varyantı.

    Args:
        input_dim:    Feature sayısı.
        hidden_dim:   GCN birinci katman çıkış boyutu (default: config.HIDDEN_DIM).
        output_dim:   GCN ikinci katman çıkış boyutu (default: config.OUTPUT_DIM).
        num_classes:  Sınıf sayısı (default: config.NUM_CLASSES = 2).
        dropout:      Dropout oranı (default: config.DROPOUT).
    """

    variant: str = "gcn"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        num_classes: int | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__()

        # Eksik argümanları config'den tamamla
        hidden_dim  = hidden_dim  if hidden_dim  is not None else config.HIDDEN_DIM
        output_dim  = output_dim  if output_dim  is not None else config.OUTPUT_DIM
        num_classes = num_classes if num_classes is not None else config.NUM_CLASSES
        dropout     = dropout     if dropout     is not None else config.DROPOUT

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_classes = num_classes

        # GCN Layers
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, output_dim)

        # MLP Classifier
        mlp_hidden = max(output_dim // 2, 16)
        self.fc1 = nn.Linear(output_dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,   # GCN yok sayar, arayüz uyumluluğu için
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:          (N, input_dim) — node features
            edge_index: (2, E) — edge listesi (PyG COO formatı)
            edge_attr:  kullanılmaz, BaseGNN arayüzü için kabul edilir

        Returns:
            logits: (N, num_classes)
        """
        # GCN Layer 1
        h = F.relu(self.gcn1(x, edge_index))
        h = self.dropout(h)

        # GCN Layer 2
        h = F.relu(self.gcn2(h, edge_index))
        h = self.dropout(h)

        # MLP Classifier
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h)

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Softmax olasılıkları (SHAP / LIME açıklanabilirlik için).

        Args:
            x:          (N, input_dim)
            edge_index: (2, E)
            edge_attr:  kullanılmaz, BaseGNN arayüzü için kabul edilir

        Returns:
            probs: (N, num_classes) — satır toplamları 1.0
        """
        with torch.no_grad():
            logits = self.forward(x, edge_index, edge_attr)
            return F.softmax(logits, dim=-1)


# ===========================================================================
# Yardımcı fonksiyonlar
# ===========================================================================

def count_parameters(model: nn.Module) -> int:
    """Modelin toplam eğitilebilir parametre sayısını döndür."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: ELGNN) -> None:
    """Model mimarisini ve parametre sayısını stdout'a yazdır."""
    print(f"{'=' * 70}")
    print(f"EL-GNN MODEL SUMMARY (variant: {model.variant})")
    print(f"{'=' * 70}")
    print(f"  Input dimension:      {model.input_dim}")
    print(f"  GCN hidden dimension: {model.hidden_dim}")
    print(f"  GCN output dimension: {model.output_dim}")
    print(f"  Output classes:       {model.num_classes}")
    print(f"\nLayer breakdown:")
    for name, param in model.named_parameters():
        print(
            f"  {name:40s} {str(list(param.shape)):20s} "
            f"{param.numel():>10,} params"
        )
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")


# ===========================================================================
# Smoke test
# ===========================================================================
if __name__ == "__main__":
    print("Testing ELGNN model (pure GCN variant) construction and forward pass...\n")

    INPUT_DIM = 44
    N_NODES = 1000
    N_EDGES = 10000

    model = ELGNN(input_dim=INPUT_DIM)
    model_summary(model)

    x = torch.randn(N_NODES, INPUT_DIM)
    edge_index = torch.randint(0, N_NODES, (2, N_EDGES))

    print(f"\n{'=' * 70}")
    print("Testing forward pass with dummy data:")
    print(f"  Input x shape:           {tuple(x.shape)}")
    print(f"  Input edge_index shape:  {tuple(edge_index.shape)}")

    model.eval()
    logits = model(x, edge_index)
    probs = model.predict_proba(x, edge_index)

    print(f"  Output logits shape:     {tuple(logits.shape)}")
    print(f"  Output probs shape:      {tuple(probs.shape)}")
    print(f"  Probs sum (should be 1): {probs.sum(dim=-1)[:5].tolist()}")
    print(f"\n✓ Forward pass successful")
