"""
preprocessing/pipeline.py

Tek bir dataset için uçtan-uca preprocessing pipeline'ı.
Her adımı step() context manager ile zamanlayarak raporlar.

Pipeline adımları:
  1. Input DataFrame'i al (zaten yüklenmiş)
  2. Cleaning (NaN/inf/dup temizle)
  3. Kategorik encoding (label korunur)
  4. Stratified train/test split (%70/%30, multi-class label üzerinden)
  5. Binary label dönüşümü
  6. log1p transform
  7. MinMaxScaler [0, 1]
  8. SMOTE oversampling (sadece train)
  9. ENN cleanup (sadece train)

Çıktılar:
  ./not_stratified_no_mi/
    ├── data1/
    │   ├── train.csv
    │   ├── test.csv
    │   └── test_w_all_classes.csv
    ├── data1_before_smote/
    │   └── ...
    ├── data2/
    └── data2_before_smote/

CLI:
    python -m preprocessing.pipeline
"""

from __future__ import annotations

import gc
import os
import sys
import time
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .cleaner import clean, encode_categoricals_keep_label
from .io import load_raw_csv, write_csv
from .sampler import run_smote_enn
from .transforms import fit_and_scale, log1p_safe
from .splitter import build_split_datasets
from ..utils.logging import banner, step

# ===========================================================================
# Pipeline constants (preprocessing modülüne özgü, config.py'ya bağımlı değil)
# ===========================================================================
RAW_CSV: str = "./CICFlowMeter_out.csv"
OUTPUT_DIR: str = "./not_stratified_no_mi"

LABEL_COL: str = "Label"
BENIGN_VALUE: Union[str, int] = "Benign"
TEST_SIZE: float = 0.30
RANDOM_STATE: int = 42
CHUNKSIZE: int = 100_000


def run_pipeline_for_dataset(
    raw_df: pd.DataFrame,
    output_name: str,
    label_col: str = LABEL_COL,
    benign_value: Union[int, str] = BENIGN_VALUE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> None:
    """
    Önceden yüklenmiş bir DataFrame için tam preprocessing pipeline'ını çalıştır.

    Args:
        raw_df:       Ham veriyi içeren DataFrame (build_split_datasets çıktısı).
        output_name:  Çıktı klasör adı ("data1" veya "data2").
        label_col:    Label sütun adı.
        benign_value: Benign sınıf değeri (binary label için eşik).
        test_size:    Test oranı (train = 1 - test_size).
        random_state: Yeniden üretilebilirlik için seed.
    """
    banner(f"PIPELINE START: {output_name}", char="=")
    pipeline_t0 = time.time()

    df = raw_df.copy()

    # ---- Step 1: Zaten yüklenmiş ----
    with step("Already loaded (input DataFrame)", 1):
        print(f"      shape: {df.shape}", flush=True)
        print(f"      multi-class distribution:\n{df[label_col].value_counts()}", flush=True)

    # ---- Step 2: Cleaning ----
    with step("Cleaning", 2):
        df = clean(df, label_col)
        print(f"      shape: {df.shape}", flush=True)
        print(
            f"      multi-class distribution after cleaning:\n{df[label_col].value_counts()}",
            flush=True,
        )

    # ---- Step 3: Kategorik encoding (label HARİÇ) ----
    with step("Encoding categoricals (label preserved)", 3):
        df = encode_categoricals_keep_label(df, label_col)

    # Multi-class label'ı çıkar, X'i numpy'ye çevir
    print("\n      → extracting y (multi-class)...", flush=True)
    y_multiclass: np.ndarray = df[label_col].astype(str).values
    print(f"      → y_multiclass shape: {y_multiclass.shape}", flush=True)

    print("      → dropping label column...", flush=True)
    df = df.drop(columns=[label_col])
    feature_names: list[str] = df.columns.tolist()
    print(f"      → X DataFrame shape: {df.shape}", flush=True)

    print("      → converting to float32 numpy array (column-by-column)...", flush=True)
    t_conv = time.time()
    n_rows, n_cols = df.shape
    X = np.empty((n_rows, n_cols), dtype=np.float32)
    from tqdm.auto import tqdm
    for i, col in enumerate(tqdm(df.columns, desc="Converting", unit="col")):
        X[:, i] = df[col].to_numpy(dtype=np.float32, copy=False)
    print(f"      → done in {time.time() - t_conv:.1f}s", flush=True)

    del df
    gc.collect()
    print(f"      → X: {X.shape} {X.dtype}  ({X.nbytes / 1e9:.2f} GB)", flush=True)

    # ---- Step 4: Stratified split ----
    with step("Train/test split 70/30 (stratified ON MULTI-CLASS label)", 4):
        X_train, X_test, y_train_mc, y_test_mc = train_test_split(
            X, y_multiclass,
            test_size=test_size,
            stratify=y_multiclass,
            random_state=random_state,
        )
        del X, y_multiclass
        gc.collect()
        print(f"      train: {X_train.shape}  |  test: {X_test.shape}", flush=True)
        print(f"      train class dist:\n{pd.Series(y_train_mc).value_counts()}", flush=True)
        print(f"      test  class dist:\n{pd.Series(y_test_mc).value_counts()}", flush=True)

    # ---- Step 5: Binary label dönüşümü ----
    with step("Convert labels to binary (Benign=0, attack=1)", 5):
        y_train = (y_train_mc != benign_value).astype(np.int8)
        y_test = (y_test_mc != benign_value).astype(np.int8)
        print(
            f"      train binary dist: {pd.Series(y_train).value_counts().to_dict()}",
            flush=True,
        )
        print(
            f"      test  binary dist: {pd.Series(y_test).value_counts().to_dict()}",
            flush=True,
        )

    # ---- Step 6: log1p ----
    with step("log1p (in-place)", 6):
        X_train = log1p_safe(X_train)
        X_test = log1p_safe(X_test)

    # ---- Step 7: MinMaxScaler ----
    with step("MinMaxScaler [0, 1] (fit on train only)", 7):
        X_train, X_test, _ = fit_and_scale(X_train, X_test)

    # MI feature selection YOK — tüm feature'lar korunur
    selected_features = feature_names
    print(
        f"\n      → keeping all {len(selected_features)} features (no MI selection)",
        flush=True,
    )

    # ---- Checkpoint: SMOTE öncesi kaydet ----
    pre_smote_dir = os.path.join(OUTPUT_DIR, f"{output_name}_before_smote")
    banner(f"CHECKPOINT SAVE → {pre_smote_dir}", char="-")
    write_csv(X_train, y_train, selected_features,
              os.path.join(pre_smote_dir, "train.csv"))
    write_csv(X_test, y_test, selected_features,
              os.path.join(pre_smote_dir, "test.csv"))
    write_csv(X_test, y_test_mc, selected_features,
              os.path.join(pre_smote_dir, "test_w_all_classes.csv"))

    # ---- Step 8+9: SMOTE oversampling → ENN cleanup (sadece train) ----
    # run_smote_enn her iki adımı sırayla uygular ve her birinin öncesi/sonrası
    # dağılımını raporlar.
    with step("SMOTE oversampling + ENN cleanup (train only)", 8):
        X_train, y_train = run_smote_enn(X_train, y_train)

    # ---- Final save ----
    final_dir = os.path.join(OUTPUT_DIR, output_name)
    banner(f"FINAL SAVE → {final_dir}", char="-")
    write_csv(X_train, y_train, selected_features,
              os.path.join(final_dir, "train.csv"))
    write_csv(X_test, y_test, selected_features,
              os.path.join(final_dir, "test.csv"))
    write_csv(X_test, y_test_mc, selected_features,
              os.path.join(final_dir, "test_w_all_classes.csv"))

    total = time.time() - pipeline_t0
    mins, secs = divmod(total, 60)
    banner(f"PIPELINE DONE: {output_name}  ({int(mins)}m {secs:.1f}s)", char="=")


# ===========================================================================
# CLI
# ===========================================================================
if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)   # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)   # type: ignore[attr-defined]

    overall_t0 = time.time()
    print(f"\nLoading raw CSV: {RAW_CSV}", flush=True)
    print(f"Output directory: {OUTPUT_DIR}\n", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_df = load_raw_csv(RAW_CSV, chunksize=CHUNKSIZE)
    data1_df, data2_df = build_split_datasets(raw_df)

    del raw_df
    gc.collect()

    for df, name in [(data1_df, "data1"), (data2_df, "data2")]:
        try:
            run_pipeline_for_dataset(raw_df=df, output_name=name)
        except Exception as exc:
            print(f"\n[!] PIPELINE FAILED for {name}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            print("\n  Continuing with next dataset...\n", flush=True)

    total = time.time() - overall_t0
    mins, secs = divmod(total, 60)
    banner(f"ALL DATASETS DONE  ({int(mins)}m {secs:.1f}s)", char="=")
