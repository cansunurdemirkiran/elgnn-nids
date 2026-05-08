"""
config/paths.py

Dosya sistemi sabitleri — tüm path'ler ve CSV dosya isimleri tek yerde.
Herhangi bir path değişince sadece burası güncellenir.
"""

from pathlib import Path

# ==================================================================
# DIRECTORY ROOTS
# ==================================================================
DATA_ROOT: Path = Path("not_stratified_no_mi")
OUTPUT_ROOT: Path = Path("outputs")          # model checkpoint'leri, plotlar, loglar

# ==================================================================
# DATASET NAMES
# ==================================================================
DATASETS: list[str] = ["data1", "data2"]

# ==================================================================
# CSV FILE NAMES (her dataset için ortak şema)
# ==================================================================
CSV_FILES: dict[str, str] = {
    "train": "train.csv",
    "test": "test.csv",
    "test_multi": "test_w_all_classes.csv",  # multi-class label — SHAP per-attack-type için
}

# ==================================================================
# COLUMN NAMES
# ==================================================================
LABEL_COLUMN: str = "Label"
