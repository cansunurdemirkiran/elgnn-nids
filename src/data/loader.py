"""
data/loader.py

CSV dosyalarını yükler, drop'ları uygular ve tip güvenli split nesneleri döndürür.

Split mantığı:
  train.csv              → TrainSplit  (SMOTEENN sonrası, binary label)
  test_w_all_classes.csv → %30 val + %70 test olarak stratified split:
      Her ikisi de EvalSplit döndürür (binary + multi-class label).

Public API:
    load_dataset(dataset_name, split, verbose) → TrainSplit | EvalSplit
    load_all_datasets(verbose)                 → DatasetCollection
    sanity_check(verbose)                      → bool
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.config import paths as config
from .schema import DatasetCollection, EvalSplit, TrainSplit
from ..preprocessing.transforms import apply_drops, encode_binary_labels, to_tensors

# ===========================================================================
# Split constants
# ===========================================================================
TRAIN_CSV: str = "train.csv"
TEST_MULTI_CSV: str = "test_w_all_classes.csv"

VAL_RATIO: float = 0.30           # %30 validation, %70 test
SPLIT_RANDOM_STATE: int = 42      # deterministic split


def _load_csv(dataset_name: str, csv_filename: str) -> pd.DataFrame:
    """Tek bir CSV dosyasını yükle."""
    path = config.DATA_ROOT / dataset_name / csv_filename
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _split_features_label(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    DataFrame'i feature matrix ve label vektörüne ayır.

    Raises:
        ValueError: Label kolonu bulunamazsa.
    """
    label_col = config.LABEL_COLUMN
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found.\n"
            f"Available: {df.columns.tolist()}"
        )
    y = df[label_col].values
    X = df.drop(columns=[label_col])
    return X, y


def load_train(dataset_name: str, verbose: bool = False) -> TrainSplit:
    """
    train.csv'yi yükleyip TrainSplit döndür.

    Args:
        dataset_name: "data1" veya "data2"
        verbose:      Yükleme detaylarını yazdır.

    Returns:
        TrainSplit(X, y, feature_names)
    """
    if verbose:
        print(f"\nLoading {dataset_name}/train...")

    df = _load_csv(dataset_name, TRAIN_CSV)
    if verbose:
        print(f"  Initial shape: {df.shape}")

    df = apply_drops(df, verbose=verbose)
    if verbose:
        print(f"  After drops:   {df.shape}")

    X, y_raw = _split_features_label(df)
    feature_names = list(X.columns)
    y_binary = encode_binary_labels(y_raw)
    X_tensor, y_tensor = to_tensors(X, y_binary)

    if verbose:
        n_attack = (y_binary == 1).sum()
        n_benign = (y_binary == 0).sum()
        print(f"  Class distribution: benign={n_benign:,}  attack={n_attack:,}")
        print(f"  Feature count:      {len(feature_names)}")

    return TrainSplit(X=X_tensor, y=y_tensor, feature_names=feature_names)


def load_val_and_test(
    dataset_name: str,
    verbose: bool = False,
) -> tuple[EvalSplit, EvalSplit]:
    """
    test_w_all_classes.csv'yi multi-class label'a göre stratified olarak
    %30 validation / %70 test'e böl.

    Args:
        dataset_name: "data1" veya "data2"
        verbose:      Yükleme detaylarını yazdır.

    Returns:
        (val_split, test_split) — her ikisi EvalSplit.
    """
    if verbose:
        print(
            f"\nLoading {dataset_name}/test_w_all_classes "
            "(will split val/test)..."
        )

    df = _load_csv(dataset_name, TEST_MULTI_CSV)
    if verbose:
        print(f"  Initial shape: {df.shape}")

    df = apply_drops(df, verbose=verbose)
    if verbose:
        print(f"  After drops:   {df.shape}")

    X, y_multi_raw = _split_features_label(df)
    feature_names = list(X.columns)

    # Multi-class label üzerinden stratified split
    indices = np.arange(len(df))
    val_idx, test_idx = train_test_split(
        indices,
        test_size=(1 - VAL_RATIO),   # 0.70 test
        stratify=y_multi_raw,
        random_state=SPLIT_RANDOM_STATE,
    )

    if verbose:
        print(
            f"  Stratified split: val={len(val_idx):,} ({VAL_RATIO*100:.0f}%), "
            f"test={len(test_idx):,} ({(1-VAL_RATIO)*100:.0f}%)"
        )

    def _build_eval_split(idx: np.ndarray) -> EvalSplit:
        X_part = X.iloc[idx].reset_index(drop=True)
        y_multi = y_multi_raw[idx]
        y_bin = encode_binary_labels(y_multi)
        X_tensor, y_tensor = to_tensors(X_part, y_bin)
        return EvalSplit(
            X=X_tensor,
            y_binary=y_tensor,
            feature_names=feature_names,
            y_multi=y_multi,
        )

    val_split = _build_eval_split(val_idx)
    test_split = _build_eval_split(test_idx)

    if verbose:
        for name, split in [("Validation", val_split), ("Test", test_split)]:
            print(f"\n  {name} class distribution:")
            unique, counts = np.unique(split.y_multi, return_counts=True)
            for cls, cnt in zip(unique, counts):
                print(f"    {str(cls):<20s} {cnt:>6,}")

    return val_split, test_split


def load_dataset(
    dataset_name: str,
    split: str,
    verbose: bool = False,
) -> TrainSplit | EvalSplit:
    """
    Tek bir dataset/split döndür.

    Args:
        dataset_name: "data1" veya "data2"
        split:        "train" | "val" | "test"
        verbose:      Yükleme detaylarını yazdır.

    Returns:
        TrainSplit eğer split == "train",
        EvalSplit  eğer split == "val" veya "test".

    Raises:
        ValueError: Bilinmeyen split adı verilirse.
    """
    if split == "train":
        return load_train(dataset_name, verbose=verbose)

    if split in ("val", "test"):
        val_data, test_data = load_val_and_test(dataset_name, verbose=verbose)
        return val_data if split == "val" else test_data

    raise ValueError(f"Unknown split '{split}'. Use 'train', 'val', or 'test'.")


def load_all_datasets(verbose: bool = False) -> DatasetCollection:
    """
    Tüm dataset'lerin tüm split'lerini yükle.

    Returns:
        {
            "data1": {
                "train": TrainSplit,
                "val":   EvalSplit,
                "test":  EvalSplit,
            },
            "data2": { ... },
        }
    """
    result: DatasetCollection = {}
    for ds in config.DATASETS:
        result[ds] = {}
        result[ds]["train"] = load_train(ds, verbose=verbose)
        val_data, test_data = load_val_and_test(ds, verbose=verbose)
        result[ds]["val"] = val_data
        result[ds]["test"] = test_data
    return result


def sanity_check(verbose: bool = True) -> bool:
    """
    Tüm split'leri yükle, feature alignment ve sınıf dağılımı kontrol et.

    Returns:
        True: Tüm kontroller geçildiyse.
        False: Feature mismatch varsa.
    """
    print("=" * 70)
    print("SANITY CHECK: Loading all datasets and verifying alignment")
    print("=" * 70)

    all_data = load_all_datasets(verbose=verbose)

    # Feature alignment kontrolü
    reference_features: list[str] | None = None
    for ds_name, splits in all_data.items():
        for split_name, data in splits.items():
            features = data.feature_names
            if reference_features is None:
                reference_features = features
                print(
                    f"\nReference feature set ({len(features)} features) "
                    f"from {ds_name}/{split_name}"
                )
            elif features != reference_features:
                print(f"\n⚠️  MISMATCH in {ds_name}/{split_name}!")
                return False

    print(f"\n✓ All splits have identical feature space ({len(reference_features)} features)")

    # Özet tablo
    print(
        f"\n{'Dataset':<10s} {'Split':<8s} {'Samples':>10s} {'Benign':>10s} "
        f"{'Attack':>10s} {'Ratio':>8s}"
    )
    print("-" * 70)
    for ds_name, splits in all_data.items():
        for split_name, data in splits.items():
            y = data.y if isinstance(data, TrainSplit) else data.y_binary
            n_total = len(y)
            n_attack = (y == 1).sum().item()
            n_benign = (y == 0).sum().item()
            ratio = n_attack / max(n_benign, 1)
            print(
                f"{ds_name:<10s} {split_name:<8s} {n_total:>10,} "
                f"{n_benign:>10,} {n_attack:>10,} {ratio:>7.2f}:1"
            )

    print(f"\n{'=' * 70}")
    print("Feature list:")
    print(f"{'=' * 70}")
    for i, f in enumerate(reference_features, start=1):
        print(f"  {i:2d}. {f}")

    return True


if __name__ == "__main__":
    sanity_check(verbose=True)
