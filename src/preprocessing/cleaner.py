"""
preprocessing/cleaner.py

Ham DataFrame üzerinde veri kalitesi işlemleri:
  - NaN / Inf temizleme
  - Duplicate satır kaldırma
  - Kategorik feature encoding (label korunur)

Public API:
    clean(df, label_col)                          → pd.DataFrame
    encode_categoricals_keep_label(df, label_col) → pd.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm


def clean(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Temel veri kalitesi adımları:
      1. Inf değerleri NaN'a çevir, NaN satırlarını at.
      2. Duplicate satırları at.
      3. Label kolonunu string ise strip et.

    Args:
        df:        İşlenecek DataFrame.
        label_col: Label sütun adı (drop edilmez, sadece strip edilir).

    Returns:
        Temizlenmiş, reset index'li DataFrame.
    """
    n_before_nan = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"      dropped NaN/inf: {n_before_nan - len(df):,}", flush=True)

    n_before_dup = len(df)
    df = df.drop_duplicates()
    print(f"      dropped duplicates: {n_before_dup - len(df):,}", flush=True)

    if label_col in df.columns and df[label_col].dtype == object:
        df[label_col] = df[label_col].astype(str).str.strip()

    return df.reset_index(drop=True)


def encode_categoricals_keep_label(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Label kolonu hariç tüm object-dtype feature'ları LabelEncoder ile encode et.

    Label kolonu dokunulmadan korunur; multi-class string label'ın
    sonraki adımlarda binary'ye çevrilebilmesi için gereklidir.

    Args:
        df:        İşlenecek DataFrame.
        label_col: Korunacak label sütun adı.

    Returns:
        Kategorik feature'ları sayısal yapılmış DataFrame.
    """
    object_cols = [
        c for c in df.select_dtypes(include=["object"]).columns
        if c != label_col
    ]

    if not object_cols:
        print("      no object feature columns to encode", flush=True)
        return df

    for col in tqdm(object_cols, desc="Encoding", unit="col"):
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df
