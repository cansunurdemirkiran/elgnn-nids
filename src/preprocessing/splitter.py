"""
preprocessing/splitter.py

Ham DataFrame'i data1 ve data2 alt setlerine böler.
  - Benign satırları attack oranına göre orantılı dağıtır.
  - Her dataset için karıştırma (shuffle) uygulanır.

Public API:
    build_split_datasets(raw_df, random_state) → (data1_df, data2_df)
"""

from __future__ import annotations

import pandas as pd
from ..utils.logging import banner

# Attack türü → dataset eşlemesi
ATTACK_SPLIT: dict[str, list[str]] = {
    "data1": ["Reconnaissance", "Fuzzers", "Worms", "Analysis"],
    "data2": ["Backdoor", "DoS", "Exploits", "Generic", "Shellcode"],
}

TOTAL_BENIGN: int = 30_000  # Reduced for sample data - original was 300_000
BENIGN_VALUE: str = "Benign"
LABEL_COL: str = "Label"


def build_split_datasets(
    raw_df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ham veriden data1 ve data2'yi oluştur.

    Adımlar:
      1. Benign ve attack satırlarını ayır.
      2. Attack'ı ATTACK_SPLIT tanımına göre iki gruba böl.
      3. Benign'i attack oranına göre orantılı seç ve dağıt.
      4. Her iki seti karıştır.

    Args:
        raw_df:       Ham CICFlowMeter CSV'sinden yüklenen DataFrame.
        random_state: Yeniden üretilebilirlik için random state.

    Returns:
        (data1_df, data2_df) — karıştırılmış, reset index'li DataFrames.

    Raises:
        ValueError: İstenen benign sayısı mevcut benign'den fazlaysa.
    """
    banner("BUILDING data1 / data2 FROM RAW", char="=")

    benign_mask = raw_df[LABEL_COL] == BENIGN_VALUE
    benign_df = raw_df[benign_mask]
    attack_df = raw_df[~benign_mask]

    print(f"  Total raw: {len(raw_df):,}", flush=True)
    print(f"    Benign: {len(benign_df):,}", flush=True)
    print(f"    Attack: {len(attack_df):,}", flush=True)
    print(f"    Attack distribution:", flush=True)
    for cls, n in attack_df[LABEL_COL].value_counts().items():
        print(f"      {cls:<20s} {n:>8,}", flush=True)

    # Attack'i data1 ve data2'ye böl
    data1_attack = attack_df[attack_df[LABEL_COL].isin(ATTACK_SPLIT["data1"])].copy()
    data2_attack = attack_df[attack_df[LABEL_COL].isin(ATTACK_SPLIT["data2"])].copy()

    n1_attack = len(data1_attack)
    n2_attack = len(data2_attack)
    total_attack = n1_attack + n2_attack

    print(f"\n  data1 attacks: {n1_attack:,}  ({ATTACK_SPLIT['data1']})", flush=True)
    print(f"  data2 attacks: {n2_attack:,}  ({ATTACK_SPLIT['data2']})", flush=True)
    print(
        f"  ratio: data1={n1_attack/total_attack:.3f}  "
        f"data2={n2_attack/total_attack:.3f}",
        flush=True,
    )

    # Benign'i attack oranına göre orantılı böl
    n1_benign = int(round(TOTAL_BENIGN * n1_attack / total_attack))
    n2_benign = TOTAL_BENIGN - n1_benign  # yuvarlama hatasını absorbe eder
    print(
        f"\n  benign allocation: data1={n1_benign:,}  data2={n2_benign:,}"
        f"  (total={TOTAL_BENIGN:,})",
        flush=True,
    )

    if n1_benign + n2_benign > len(benign_df):
        raise ValueError(
            f"İstenen toplam benign ({n1_benign + n2_benign:,}) "
            f"mevcut benign sayısından ({len(benign_df):,}) fazla."
        )

    # Benign rastgele seç ve böl
    benign_shuffled = benign_df.sample(
        n=n1_benign + n2_benign,
        random_state=random_state,
    ).reset_index(drop=True)
    data1_benign = benign_shuffled.iloc[:n1_benign]
    data2_benign = benign_shuffled.iloc[n1_benign:]

    # Birleştir ve karıştır
    data1_df = (
        pd.concat([data1_benign, data1_attack], ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    data2_df = (
        pd.concat([data2_benign, data2_attack], ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    print(f"\n  → data1: {data1_df.shape}", flush=True)
    print(f"      class dist:\n{data1_df[LABEL_COL].value_counts().to_string()}", flush=True)
    print(f"\n  → data2: {data2_df.shape}", flush=True)
    print(f"      class dist:\n{data2_df[LABEL_COL].value_counts().to_string()}", flush=True)

    return data1_df, data2_df
