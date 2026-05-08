"""
models/e_graphsage.py

E-GraphSAGE — Edge-feature destekli GraphSAGE tabanlı IDS modeli.

Referans:
  Lo et al. (2021) "E-GraphSAGE: A Graph Neural Network based Intrusion
  Detection System for IoT" — NDSS Symposium 2022.

GCN'den farkı:
  - Spektral convolution (GCNConv) yerine inductive aggregation (SAGEConv)
    → test sırasında görülmemiş node'lara genelleyebilir
  - Edge feature'ları aggregation'a dahil eder
    → ağ akışları arasındaki ilişki bilgisi de öğrenilir
  - Her edge (u→v) için: edge_feat = |x_u - x_v|  (özellik farkı)
    → benzer akışları ve farklı akışları ayırt edebilir

Mimari:
  Input (N, F)
      ↓
  EGraphSAGEConv Layer 1  (F, F → H)  + ReLU + Dropout
      ↓
  EGraphSAGEConv Layer 2  (H, F → O)  + ReLU + Dropout
      ↓
  Linear (O → O/2) + ReLU + Dropout
      ↓
  Linear (O/2 → num_classes)

EGraphSAGEConv içi mesaj geçişi:
  message(u→v):  concat(h_u, edge_feat_{uv})  → lin_msg → h_msg
  aggregate:     mean({h_msg for u in N(v)})
  update(v):     concat(lin_self(h_v), agg)   → lin_out → h_v'

Public API:
    EGraphSAGE(input_dim, ...)
    count_parameters(model)  → int
    model_summary(model)     → None
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops
except ImportError:
    raise ImportError(
        "torch-geometric is not installed. Install it with:\n"
        "  pip install torch-geometric"
    )

from src.config import hyperparams as config
from .base import BaseGNN


# ===========================================================================
# Custom MessagePassing katmanı
# ===========================================================================

class EGraphSAGEConv(MessagePassing):
    """
    Edge-feature destekli GraphSAGE convolution katmanı.

    Her mesajda kaynak node feature'ı ile edge feature'ı birleştirilir;
    bu sayede sadece komşu varlığı değil, komşu ile ilişki de öğrenilir.

    Args:
        in_channels:  Girdi node feature boyutu.
        out_channels: Çıktı node feature boyutu.
        edge_dim:     Edge feature boyutu (genellikle == in_channels).
        bias:         Linear katmanlara bias eklensin mi.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__(aggr="mean")   # mean aggregation (E-GraphSAGE paper)

        # Komşu mesajı: concat(h_u, edge_feat) → out_channels
        self.lin_msg = nn.Linear(in_channels + edge_dim, out_channels, bias=bias)

        # Self-loop projeksiyon
        self.lin_self = nn.Linear(in_channels, out_channels, bias=bias)

        # Birleştirme: concat(self_proj, agg_msg) → out_channels
        self.lin_out = nn.Linear(out_channels * 2, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin_msg.weight)
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        """
        Args:
            x:          (N, in_channels)
            edge_index: (2, E)
            edge_attr:  (E, edge_dim)

        Returns:
            (N, out_channels)
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        """
        Kaynak node (j) feature'ı ile edge feature'ını birleştirip dönüştür.

        Args:
            x_j:       (E, in_channels) — kaynak node feature'ları
            edge_attr: (E, edge_dim)

        Returns:
            (E, out_channels)
        """
        msg_input = torch.cat([x_j, edge_attr], dim=-1)   # (E, in_ch + edge_dim)
        return F.relu(self.lin_msg(msg_input))

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        """
        Aggregate edilmiş mesajları self-loop ile birleştir.

        Args:
            aggr_out: (N, out_channels) — ortalama mesaj
            x:        (N, in_channels) — mevcut node feature

        Returns:
            (N, out_channels)
        """
        self_proj = self.lin_self(x)                          # (N, out_channels)
        combined = torch.cat([self_proj, aggr_out], dim=-1)   # (N, out_channels*2)
        return self.lin_out(combined)


# ===========================================================================
# E-GraphSAGE modeli
# ===========================================================================

class EGraphSAGE(BaseGNN):
    """
    E-GraphSAGE — Edge-feature destekli GraphSAGE IDS modeli.

    Edge feature'ları model dışında hesaplanıp verilebilir (graph_builder)
    ya da forward içinde düğüm farklarından otomatik türetilebilir.

    Args:
        input_dim:   Node feature sayısı (F).
        hidden_dim:  Birinci SAGE katman çıkışı (default: config.HIDDEN_DIM).
        output_dim:  İkinci SAGE katman çıkışı (default: config.OUTPUT_DIM).
        num_classes: Sınıf sayısı (default: config.NUM_CLASSES).
        dropout:     Dropout oranı (default: config.DROPOUT).
        edge_dim:    Edge feature boyutu; None ise input_dim kullanılır.
                     Graph builder edge_attr sağlamıyorsa otomatik hesaplanır.
    """

    variant: str = "e_graphsage"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        num_classes: int | None = None,
        dropout: float | None = None,
        edge_dim: int | None = None,
    ) -> None:
        super().__init__()

        hidden_dim  = hidden_dim  if hidden_dim  is not None else config.HIDDEN_DIM
        output_dim  = output_dim  if output_dim  is not None else config.OUTPUT_DIM
        num_classes = num_classes if num_classes is not None else config.NUM_CLASSES
        dropout     = dropout     if dropout     is not None else config.DROPOUT
        edge_dim    = edge_dim    if edge_dim    is not None else input_dim

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.edge_dim   = edge_dim

        # E-GraphSAGE Katmanları
        self.sage1 = EGraphSAGEConv(input_dim,  hidden_dim, edge_dim=edge_dim)
        self.sage2 = EGraphSAGEConv(hidden_dim, output_dim, edge_dim=edge_dim)

        # MLP Classifier
        mlp_hidden = max(output_dim // 2, 16)
        self.fc1 = nn.Linear(output_dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, num_classes)

        self.dropout = nn.Dropout(dropout)

    def _compute_edge_attr(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Edge feature'ı node fark vektörü olarak hesapla.

        edge_attr[i] = |x[src_i] - x[dst_i]|

        Bu yöntem graph_builder edge_attr sağlamadığında kullanılır.
        Absolüt fark, iki akış arasındaki farklılığı encode eder.

        Args:
            x:          (N, F) — node feature matrisi
            edge_index: (2, E)

        Returns:
            (E, F) — edge feature matrisi
        """
        src, dst = edge_index[0], edge_index[1]
        return torch.abs(x[src] - x[dst])

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x:          (N, input_dim) — node features
            edge_index: (2, E)
            edge_attr:  (E, edge_dim) — opsiyonel; None ise |x_src - x_dst| kullanılır

        Returns:
            logits: (N, num_classes)
        """
        # Edge feature yoksa otomatik hesapla
        if edge_attr is None:
            edge_attr = self._compute_edge_attr(x, edge_index)

        # E-GraphSAGE Layer 1
        h = self.sage1(x, edge_index, edge_attr)
        h = F.relu(h)
        h = self.dropout(h)

        # E-GraphSAGE Layer 2 — edge_attr boyutu Layer 1 çıkışıyla uyumsuz
        # olabileceği için orijinal edge_attr'ı kullanmaya devam ediyoruz
        h = self.sage2(h, edge_index, edge_attr)
        h = F.relu(h)
        h = self.dropout(h)

        # MLP Classifier
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h)

    def predict_proba(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """
        Softmax olasılıkları — SHAP açıklanabilirlik için.

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
    """Toplam eğitilebilir parametre sayısını döndür."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: EGraphSAGE) -> None:
    """Model mimarisini ve parametre sayısını stdout'a yazdır."""
    print(f"{'=' * 70}")
    print(f"E-GraphSAGE MODEL SUMMARY (variant: {model.variant})")
    print(f"{'=' * 70}")
    print(f"  Input dimension:        {model.input_dim}")
    print(f"  Edge feature dimension: {model.edge_dim}")
    print(f"  SAGE hidden dimension:  {model.hidden_dim}")
    print(f"  SAGE output dimension:  {model.output_dim}")
    print(f"  Output classes:         {model.num_classes}")
    print(f"\nLayer breakdown:")
    for name, param in model.named_parameters():
        print(
            f"  {name:45s} {str(list(param.shape)):20s} "
            f"{param.numel():>10,} params"
        )
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")


# ===========================================================================
# Smoke test
# ===========================================================================
if __name__ == "__main__":
    print("Testing EGraphSAGE construction and forward pass...\n")

    INPUT_DIM = 44
    N_NODES = 1000
    N_EDGES = 10000

    model = EGraphSAGE(input_dim=INPUT_DIM)
    model_summary(model)

    x = torch.randn(N_NODES, INPUT_DIM)
    edge_index = torch.randint(0, N_NODES, (2, N_EDGES))

    print(f"\n{'=' * 70}")
    print("Test 1: forward WITHOUT edge_attr (auto-computed)")
    model.eval()
    logits = model(x, edge_index)
    probs = model.predict_proba(x, edge_index)
    print(f"  logits shape: {tuple(logits.shape)}")
    print(f"  probs  shape: {tuple(probs.shape)}")
    print(f"  probs sum:    {probs.sum(dim=-1)[:5].tolist()}")

    print(f"\nTest 2: forward WITH explicit edge_attr")
    edge_attr = torch.randn(N_EDGES, INPUT_DIM)
    logits2 = model(x, edge_index, edge_attr)
    print(f"  logits shape: {tuple(logits2.shape)}")
    print(f"\n✓ Both forward passes successful")
