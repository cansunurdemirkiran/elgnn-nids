"""
config/hyperparams.py

Model ve training hyperparameter sabitleri.
Buradan bir değer değişince tüm pipeline etkilenir.

Projede karşılaştırılan üç senaryo:
  Scenario A — GCN (baseline):       ELGNN, EWC yok, Task B'de eğitilir
  Scenario B — GCN + EWC:            ELGNN, EWC ile Task A→B continual learning
  Scenario C — E-GraphSAGE + EWC:    EGraphSAGE, EWC ile Task A→B continual learning
"""

# ==================================================================
# GRAPH CONSTRUCTION
# ==================================================================
KNN_K: int = 10          # her node için en yakın k komşu

# ==================================================================
# GCN HYPERPARAMETERS  (Scenario A & B)
# ==================================================================
HIDDEN_DIM: int = 128
OUTPUT_DIM: int = 64     # GCN'in son layer çıkış boyutu
NUM_ATTENTION_HEADS: int = 4
DROPOUT: float = 0.3
NUM_CLASSES: int = 2     # binary: benign vs attack

# ==================================================================
# E-GraphSAGE HYPERPARAMETERS  (Scenario C)
# ==================================================================
SAGE_HIDDEN_DIM: int = 128   # SAGE birinci katman çıkışı
SAGE_OUTPUT_DIM: int = 64    # SAGE ikinci katman çıkışı
SAGE_DROPOUT: float = 0.3
# SAGE_EDGE_DIM: None → input_dim ile aynı (otomatik ayarlanır)

# ==================================================================
# TRAINING HYPERPARAMETERS  (tüm modeller için ortak)
# ==================================================================
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
NUM_EPOCHS_TASK_A: int = 100
NUM_EPOCHS_TASK_B: int = 100
BATCH_SIZE: int = 128    # mini-batch sampling için (graph batching)
EARLY_STOPPING_PATIENCE: int = 15

# ==================================================================
# EWC (Elastic Weight Consolidation)  — Scenario B & C
# ==================================================================
EWC_LAMBDA: float = 100       # EL-GNN paper'ı 100-150 öneriyor
FISHER_SAMPLES: int = 1000    # Fisher hesabı için kaç sample kullanılacak

# ==================================================================
# CLASS WEIGHT
# ==================================================================
USE_CLASS_WEIGHT: bool = False  # SMOTEENN balanced yaptıysa False

# ==================================================================
# REPRODUCIBILITY
# ==================================================================
RANDOM_SEED: int = 42

# ==================================================================
# EXPERIMENT SCENARIO NAMES  (checkpoint dosya adlarında kullanılır)
# ==================================================================
SCENARIO_GCN: str = "gcn_baseline"
SCENARIO_GCN_EWC: str = "gcn_ewc"
SCENARIO_EGRAPHSAGE_EWC: str = "e_graphsage_ewc"
