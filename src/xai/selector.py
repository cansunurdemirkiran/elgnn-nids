"""
xai/selector.py

Her test seti için SHAP analizinde kullanılacak sample'ları seçer.

Seçim mantığı:
  - Senaryo 2 (Task B alone) modelini yükle.
  - Her test seti için: modelin DOĞRU tahmin ettiği sample'ları filtrele.
  - Her saldırı türünden: N_BACKGROUND_PER_ATTACK background + N_EXPLAINED_PER_ATTACK explained.
  - Benign'den:          (saldırı türü × N_BACKGROUND_PER_ATTACK) background + N_BENIGN_EXPLAINED explained.
  - Sonuçları JSON olarak kaydet.

Çıktı:
  xai_outputs/data1/selected_samples.json
  xai_outputs/data2/selected_samples.json

CLI:
    python -m xai.selector
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import hyperparams as config
from ..models.elgnn import ELGNN
from .schema import SampleSelection

# ===========================================================================
# Seçim sabitleri
# ===========================================================================
MODEL_NAME: str = "gcn_taskB_alone_h128_o64_k10_lr1e-03_lambda100.pt"
MODEL_PATH: Path = Path("models") / MODEL_NAME
OUTPUT_ROOT: Path = Path("xai_outputs")

N_BACKGROUND_PER_ATTACK: int = 5
N_EXPLAINED_PER_ATTACK: int = 2
N_BENIGN_EXPLAINED: int = 10

RANDOM_SEED: int = 42


# ===========================================================================
# İç yardımcılar
# ===========================================================================

def _get_model_predictions(
    model: ELGNN,
    graph_data,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Modelin tüm node'lar için tahminlerini ve olasılıklarını döndür.

    Args:
        model:      Değerlendirilecek model.
        graph_data: PyG Data nesnesi (.x ve .edge_index içerir).
        device:     "cuda" veya "cpu".

    Returns:
        (preds, probs) — numpy arrays, shape (N,) ve (N, num_classes).
    """
    model.eval()
    x = graph_data.x.to(device)
    edge_index = graph_data.edge_index.to(device)

    with torch.no_grad():
        logits = model(x, edge_index)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

    return preds.cpu().numpy(), probs.cpu().numpy()


def _select_class_samples(
    cls: str,
    class_mask: np.ndarray,
    correct_mask: np.ndarray,
    rng: np.random.RandomState,
    n_background: int,
    n_explained: int,
) -> tuple[list[int], list[int]]:
    """
    Bir sınıf için background ve explained indekslerini seç.

    Kullanılabilir örnek yetmiyorsa uyarı verir ve mevcut olanı kullanır.

    Returns:
        (bg_indices, exp_indices)
    """
    available_idx = np.where(class_mask & correct_mask)[0]
    n_available = len(available_idx)
    n_needed = n_background + n_explained

    if n_available < n_needed:
        print(
            f"    {cls}: only {n_available} correct predictions, "
            f"need {n_needed} — using what's available"
        )
        n_take_bg = min(n_background, n_available)
        n_take_exp = min(n_explained, n_available - n_take_bg)
    else:
        n_take_bg = n_background
        n_take_exp = n_explained

    shuffled = rng.permutation(available_idx)
    bg_idx = shuffled[:n_take_bg].tolist()
    exp_idx = shuffled[n_take_bg:n_take_bg + n_take_exp].tolist()
    return bg_idx, exp_idx


# ===========================================================================
# Public API
# ===========================================================================

def select_samples_for_dataset(
    dataset_name: str,
    graphs: dict,
    model: ELGNN,
    device: str,
    output_dir: Path,
) -> SampleSelection:
    """
    Bir dataset için SHAP sample'larını seç ve JSON'a kaydet.

    Args:
        dataset_name: "data1" veya "data2".
        graphs:       build_all_graphs() çıktısı.
        model:        Tahmin için kullanılacak model.
        device:       "cuda" veya "cpu".
        output_dir:   JSON çıktısının kaydedileceği dizin.

    Returns:
        Seçilen sample metadata'sını taşıyan SampleSelection nesnesi.
    """
    print(f"\n{'=' * 70}")
    print(f"Selecting samples for {dataset_name}/test")
    print(f"{'=' * 70}")

    test_data = graphs[dataset_name]["test"]
    graph, feature_names, y_multi = test_data
    y_binary = graph.y.cpu().numpy()

    print(f"  Computing model predictions on {len(y_binary):,} samples...")
    preds, _ = _get_model_predictions(model, graph, device)
    correct_mask = preds == y_binary
    n_correct = correct_mask.sum()
    print(
        f"  Correct predictions: {n_correct:,} / {len(y_binary):,} "
        f"({n_correct/len(y_binary)*100:.2f}%)"
    )

    rng = np.random.RandomState(RANDOM_SEED)
    selection = SampleSelection(
        dataset=dataset_name,
        model_path=str(MODEL_PATH),
        n_total_samples=int(len(y_binary)),
        n_correct_predictions=int(n_correct),
        feature_names=feature_names,
    )

    background_indices: list[int] = []
    explained_indices: list[int] = []
    unique_classes = np.unique(y_multi)

    # ---- Saldırı sınıfları ----
    attack_classes = [
        c for c in unique_classes
        if str(c).strip().lower() not in {"benign", "normal"}
    ]

    for cls in attack_classes:
        cls_mask = y_multi == cls
        bg_idx, exp_idx = _select_class_samples(
            str(cls), cls_mask, correct_mask, rng,
            N_BACKGROUND_PER_ATTACK, N_EXPLAINED_PER_ATTACK,
        )
        background_indices.extend(bg_idx)
        explained_indices.extend(exp_idx)
        selection.background[str(cls)] = bg_idx
        selection.explained[str(cls)] = exp_idx

        print(
            f"    {str(cls):<20s}: available={cls_mask.sum():>5d}, "
            f"selected bg={len(bg_idx)} exp={len(exp_idx)}"
        )

    # ---- Benign sınıfı ----
    n_benign_background = len(attack_classes) * N_BACKGROUND_PER_ATTACK
    benign_classes = [
        c for c in unique_classes
        if str(c).strip().lower() in {"benign", "normal"}
    ]

    if not benign_classes:
        print("  WARNING: No 'Benign' class found in y_multi")
    else:
        benign_label = benign_classes[0]
        benign_mask = y_multi == benign_label
        bg_idx, exp_idx = _select_class_samples(
            str(benign_label), benign_mask, correct_mask, rng,
            n_benign_background, N_BENIGN_EXPLAINED,
        )
        background_indices.extend(bg_idx)
        explained_indices.extend(exp_idx)
        selection.background[str(benign_label)] = bg_idx
        selection.explained[str(benign_label)] = exp_idx

        print(
            f"    {str(benign_label):<20s}: available={benign_mask.sum():>5d}, "
            f"selected bg={len(bg_idx)} exp={len(exp_idx)}"
        )

    # ---- Toplam ----
    selection.n_background_total = len(background_indices)
    selection.n_explained_total = len(explained_indices)
    selection.all_background_indices = sorted(background_indices)
    selection.all_explained_indices = sorted(explained_indices)

    print(f"\n  Total background: {len(background_indices)}")
    print(f"  Total explained:  {len(explained_indices)}")

    # ---- JSON'a kaydet ----
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "selected_samples.json"
    with open(output_path, "w") as f:
        json.dump(asdict(selection), f, indent=2, default=str)
    print(f"\n  Saved: {output_path}")

    return selection


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Make sure train.py finished and the model was saved."
        )

    print(f"\nLoading model from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    model = ELGNN(input_dim=checkpoint["input_dim"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(
        f"  Loaded. Best epoch: {checkpoint['best_epoch']}, "
        f"val_loss: {checkpoint['best_val_loss']:.4f}"
    )

    print("\nLoading graphs from cache...")
    graphs = build_all_graphs(use_cache=True, verbose=False)

    OUTPUT_ROOT.mkdir(exist_ok=True)
    for dataset in ["data1", "data2"]:
        select_samples_for_dataset(
            dataset, graphs, model, device,
            output_dir=OUTPUT_ROOT / dataset,
        )

    print(f"\n\n{'=' * 70}")
    print("Done. Sample selections saved.")
    print(f"{'=' * 70}")
    print(f"  {OUTPUT_ROOT}/data1/selected_samples.json")
    print(f"  {OUTPUT_ROOT}/data2/selected_samples.json")
    print("\nNext step: explain_shap.py")


if __name__ == "__main__":
    main()
