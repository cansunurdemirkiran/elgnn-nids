"""
config/features.py

Feature mühendisliği kararları — tüm drop kategorileri ve mantıkları.
Her kategori ayrı tutulur; neden drop edildiği yorumlarla açıklanır.
get_all_drops() tek noktadan tüm listeyi döndürür.
"""

# ==================================================================
# 1. Identifier kolonlar
# k-NN distance graph kullandığımız için IP/Port'a ihtiyaç yok.
# Flow ID ve Timestamp leakage riski taşır.
# ==================================================================
IDENTIFIER_DROPS: list[str] = [
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Timestamp",
]

# ==================================================================
# 2. Zero variance — std = 0, tek değer, hiçbir bilgi taşımaz
# ==================================================================
ZERO_VARIANCE_DROPS: list[str] = [
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "URG Flag Count",
    "CWR Flag Count",
    "ECE Flag Count",
    "Fwd Bytes/Bulk Avg",
    "Fwd Packet/Bulk Avg",
    "Fwd Bulk Rate Avg",
]

# ==================================================================
# 3. Dominant value — %99+ tek değer, sinyal çok zayıf
# ==================================================================
DOMINANT_VALUE_DROPS: list[str] = [
    "Fwd PSH Flags",   # skew 49.85, %99.96 oranında 0
    "RST Flag Count",  # skew 155.96, neredeyse hiç görülmüyor
]

# ==================================================================
# 4. Exact duplicates — Pearson > 0.999, bilgi dublikasyonu
# ==================================================================
EXACT_DUPLICATE_DROPS: list[str] = [
    "Fwd Segment Size Avg",  # = Fwd Packet Length Mean (1.0 korelasyon)
    "Bwd Segment Size Avg",  # = Bwd Packet Length Mean (1.0 korelasyon)
]

# ==================================================================
# 5. High correlation drops (Pearson > 0.95 in BOTH datasets)
# Her grup için en "temel" / en yorumlanabilir feature tutulur,
# türetilmişler drop edilir.
# ==================================================================
HIGH_CORRELATION_DROPS: list[str] = [
    # ----------------------------------------------------------------
    # Group: Idle period statistics
    # Idle Mean ↔ Idle Max ↔ Idle Min  (hepsi 0.999+ korelasyon)
    # KEPT: Idle Mean
    # ----------------------------------------------------------------
    "Idle Max",
    "Idle Min",

    # ----------------------------------------------------------------
    # Group: Active period statistics
    # Active Mean ↔ Active Max ↔ Active Min  (0.98+ korelasyon)
    # KEPT: Active Mean
    # ----------------------------------------------------------------
    "Active Max",
    "Active Min",

    # ----------------------------------------------------------------
    # Group: Packet length distribution stats
    # Packet Length Std ↔ Packet Length Variance  (1.0, çünkü Var = Std²)
    # Packet Length Mean ↔ Average Packet Size    (0.999, aynı şey)
    # KEPT: Packet Length Std, Packet Length Mean
    # ----------------------------------------------------------------
    "Packet Length Variance",   # = Std² aritmetik olarak
    "Average Packet Size",       # = Packet Length Mean

    # ----------------------------------------------------------------
    # Group: Flow IAT (inter-arrival time)
    # Flow IAT Mean ↔ Flow IAT Std ↔ Flow IAT Max  (0.98+)
    # Flow IAT Mean ↔ Flow Packets/s               (0.997, IAT'ın tersi)
    # KEPT: Flow IAT Mean (zaman), Flow Packets/s (rate)
    # ----------------------------------------------------------------
    "Flow IAT Std",
    "Flow IAT Max",

    # ----------------------------------------------------------------
    # Group: Forward IAT
    # Fwd IAT Mean ↔ Fwd IAT Std ↔ Fwd IAT Max ↔ Fwd IAT Total  (0.98+)
    # KEPT: Fwd IAT Mean (ortalama), Fwd IAT Total (toplam — farklı bilgi)
    # ----------------------------------------------------------------
    "Fwd IAT Std",
    "Fwd IAT Max",

    # ----------------------------------------------------------------
    # Group: Backward IAT
    # Bwd IAT Mean ↔ Bwd IAT Std ↔ Bwd IAT Max ↔ Bwd IAT Total  (0.98+)
    # KEPT: Bwd IAT Mean (ortalama), Bwd IAT Total (toplam — farklı bilgi)
    # ----------------------------------------------------------------
    "Bwd IAT Std",
    "Bwd IAT Max",

    # ----------------------------------------------------------------
    # Group: Backward packet length
    # Bwd Packet Length Max ↔ Bwd Packet Length Mean ↔ Total Length of Bwd Packet (0.98+)
    # KEPT: Bwd Packet Length Mean, Total Length of Bwd Packet
    # ----------------------------------------------------------------
    "Bwd Packet Length Max",

    # ----------------------------------------------------------------
    # Group: Forward packet length
    # Fwd Packet Length Max ↔ Fwd Packet Length Mean (0.95)
    # Fwd Packet Length Max ↔ Total Length of Fwd Packet (0.98)
    # KEPT: Fwd Packet Length Mean, Total Length of Fwd Packet
    # ----------------------------------------------------------------
    "Fwd Packet Length Max",

    # ----------------------------------------------------------------
    # Group: Init Window Bytes (FWD ↔ Bwd 0.98 korelasyon)
    # KEPT: FWD Init Win Bytes (forward yön daha temel)
    # NOT: Protocol drop edildiği için bu feature artık Protocol'ün
    #      ezberini taşımıyor, gerçek window size bilgisi taşıyor.
    # ----------------------------------------------------------------
    "Bwd Init Win Bytes",

    # ----------------------------------------------------------------
    # Group: Packet length minimums (0.99 korelasyon)
    # Fwd Packet Length Min ↔ Packet Length Min (aynı şey)
    # KEPT: Packet Length Min (genel olduğu için)
    # ----------------------------------------------------------------
    "Fwd Packet Length Min",

    # ----------------------------------------------------------------
    # Group: Header / packet count redundancy
    # ACK Flag Count ↔ Fwd Header Length (0.96)
    # Fwd Header Length ↔ Total Fwd Packet (0.96)
    # Her TCP paketi header taşır, header length packet count ile orantılı.
    # ACK flag de TCP paket sayısı ile orantılı.
    # KEPT: Total Fwd Packet (en temel sayım)
    # ----------------------------------------------------------------
    "Fwd Header Length",
    "ACK Flag Count",

    # ----------------------------------------------------------------
    # Group: Backward bulk statistics
    # Bwd Bytes/Bulk Avg ↔ Bwd Packet/Bulk Avg (0.97)
    # KEPT: Bwd Bytes/Bulk Avg (byte daha bilgilendirici)
    # ----------------------------------------------------------------
    "Bwd Packet/Bulk Avg",

    # ----------------------------------------------------------------
    # Standalone drop: Protocol
    # Saldırı türü analizi: tüm sınıflarda dominant protokol TCP (P=6).
    # Benign %84 TCP, saldırılar %63-100 TCP — ayırt edici sinyal yok.
    # Ayrıca Init Win Bytes ve Packet Length Min ile yüksek korelasyonluydu;
    # drop edilince bu feature'lar gerçek bilgilerini taşıyabilir.
    # ----------------------------------------------------------------
    "Protocol",
]


def get_all_drops() -> list[str]:
    """
    Tüm drop kategorilerini birleştirip tek liste olarak döndür.
    Pipeline'da apply_drops() bu fonksiyonu çağırır.
    """
    return (
        IDENTIFIER_DROPS
        + ZERO_VARIANCE_DROPS
        + DOMINANT_VALUE_DROPS
        + EXACT_DUPLICATE_DROPS
        + HIGH_CORRELATION_DROPS
    )


def get_drop_categories() -> dict[str, list[str]]:
    """
    Her kategori adını ve karşılık gelen drop listesini dict olarak döndür.
    apply_drops() fonksiyonunun verbose logging'inde kullanılır.
    """
    return {
        "identifier": IDENTIFIER_DROPS,
        "zero_variance": ZERO_VARIANCE_DROPS,
        "dominant_value": DOMINANT_VALUE_DROPS,
        "exact_duplicate": EXACT_DUPLICATE_DROPS,
        "high_correlation": HIGH_CORRELATION_DROPS,
    }
