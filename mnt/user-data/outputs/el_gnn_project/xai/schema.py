"""
xai/schema.py

XAI (açıklanabilirlik) katmanının tip güvenli veri yapıları.

SampleSelection: select_samples_for_dataset() çıktısını temsil eder.
JSON serializasyonu json.dumps(asdict(selection), default=str) ile yapılır.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SampleSelection:
    """
    Bir dataset için seçilen background ve explained sample'ların metadata'sı.

    Attributes:
        dataset:                Dataset adı ("data1" / "data2").
        model_path:             Kullanılan model dosyasının yolu.
        n_total_samples:        Test setindeki toplam örnek sayısı.
        n_correct_predictions:  Modelin doğru tahmin ettiği örnek sayısı.
        feature_names:          Feature isimleri listesi.
        background:             {sınıf_adı: [index, ...]} — SHAP background seti.
        explained:              {sınıf_adı: [index, ...]} — SHAP explained seti.
        all_background_indices: Tüm background indeksleri (sıralı).
        all_explained_indices:  Tüm explained indeksleri (sıralı).
        n_background_total:     Toplam background örnek sayısı.
        n_explained_total:      Toplam explained örnek sayısı.
    """

    dataset: str
    model_path: str
    n_total_samples: int
    n_correct_predictions: int
    feature_names: list[str]
    background: dict[str, list[int]] = field(default_factory=dict)
    explained: dict[str, list[int]] = field(default_factory=dict)
    all_background_indices: list[int] = field(default_factory=list)
    all_explained_indices: list[int] = field(default_factory=list)
    n_background_total: int = 0
    n_explained_total: int = 0
