"""
preprocessing/io.py

Ham CSV okuma ve ara/son sonuç yazma işlevleri.
Büyük dosyalar chunk-by-chunk okunup yazılır; ilerleme tqdm ile gösterilir.

Public API:
    load_raw_csv(csv_path, chunksize) → pd.DataFrame
    write_csv(X, y, columns, path, ...)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Preprocess modülü kendi sabitlerini kullanır; config bağımlılığı yoktur.
_DEFAULT_LABEL_COL: str = "Label"
_DEFAULT_CHUNKSIZE: int = 100_000
_WRITE_CHUNKSIZE: int = 50_000


def load_raw_csv(
    csv_path: str,
    chunksize: int = _DEFAULT_CHUNKSIZE,
    label_col: str = _DEFAULT_LABEL_COL,
) -> pd.DataFrame:
    """
    Büyük bir CSV'yi chunk'lar halinde oku ve birleştir.

    Sütun adlarındaki baştaki/sondaki boşlukları temizler.
    Label kolonu string ise strip eder.

    Args:
        csv_path:  Ham CSV dosya yolu.
        chunksize: Her seferinde okunacak satır sayısı.
        label_col: Label sütun adı (string temizliği için).

    Returns:
        Birleştirilmiş DataFrame.

    Raises:
        FileNotFoundError: Dosya bulunamazsa.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV bulunamadı: {csv_path}")

    print(f"      counting rows in {csv_path}...", flush=True)
    with open(csv_path, "rb") as f:
        total_rows = sum(1 for _ in f) - 1  # başlık hariç
    n_chunks = max(1, (total_rows + chunksize - 1) // chunksize)
    print(f"      total rows: {total_rows:,}  |  chunks: {n_chunks}", flush=True)

    chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(csv_path, low_memory=False, chunksize=chunksize)
    for chunk in tqdm(reader, total=n_chunks, desc="Loading raw CSV", unit="chunk"):
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df.columns = df.columns.str.strip()

    if label_col in df.columns and df[label_col].dtype == object:
        df[label_col] = df[label_col].astype(str).str.strip()

    return df


def write_csv(
    X: np.ndarray,
    y: "np.ndarray | pd.Series",
    columns: list[str],
    path: str,
    label_col: str = _DEFAULT_LABEL_COL,
    chunksize: int = _WRITE_CHUNKSIZE,
) -> None:
    """
    Feature matrisini ve label vektörünü CSV'ye yaz.

    Büyük dosyaları bellek dostu biçimde chunk-by-chunk yazar.
    Hedef dizin otomatik oluşturulur.

    Args:
        X:         (N, F) feature matrisi.
        y:         (N,) label vektörü (binary veya multi-class).
        columns:   Feature sütun isimleri (len == F).
        path:      Çıktı CSV yolu.
        label_col: Label sütun adı.
        chunksize: Yazma chunk boyutu.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(X, columns=columns)
    df[label_col] = y.values if isinstance(y, pd.Series) else y

    n = len(df)
    n_chunks = max(1, (n + chunksize - 1) // chunksize)

    with tqdm(total=n_chunks, desc=f"Writing {os.path.basename(path)}", unit="chunk") as pbar:
        for i in range(n_chunks):
            lo = i * chunksize
            hi = min((i + 1) * chunksize, n)
            mode = "w" if i == 0 else "a"
            df.iloc[lo:hi].to_csv(path, mode=mode, header=(i == 0), index=False)
            pbar.update(1)

    size_mb = os.path.getsize(path) / 1e6
    print(f"      → {path}  shape={df.shape}  ({size_mb:.1f} MB)", flush=True)
