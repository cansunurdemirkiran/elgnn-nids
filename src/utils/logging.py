"""
utils/logging.py

Pipeline genelinde kullanılan görsel yardımcılar.
  - banner()  → bölüm başlıkları
  - step()    → zamanlı step context manager
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


def banner(text: str, char: str = "=", width: int = 70) -> None:
    """İki çizgi arasında kalın bir başlık yazdır."""
    print("\n" + char * width, flush=True)
    print(f"  {text}", flush=True)
    print(char * width, flush=True)


@contextmanager
def step(label: str, idx: int, total: int = 9) -> Generator[None, None, None]:
    """
    Adım başlığı ve geçen süreyi otomatik yazdıran context manager.

    Kullanım:
        with step("Cleaning", 2, total=9):
            df = clean(df)
    """
    header = f"[{idx}/{total}] {label}"
    print(f"\n{header}", flush=True)
    print("-" * len(header), flush=True)
    t0 = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - t0
        if elapsed >= 60:
            mins, secs = divmod(elapsed, 60)
            print(f"      ⏱  {int(mins)}m {secs:.1f}s", flush=True)
        else:
            print(f"      ⏱  {elapsed:.2f}s", flush=True)
