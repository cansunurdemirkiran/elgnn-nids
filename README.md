# A Comparative Study of Graph Neural Networks for Network Intrusion Detection: Performance and Feature Importance Analysis

Bu çalışma, ağ saldırısı tespiti (IDS) probleminde üç farklı GNN mimarisini karşılaştırır:
- **GCN** (baseline)
- **GCN + EWC** (continual learning)
- **E-GraphSAGE** (edge-feature destekli, continual learning)

Karşılaştırma iki eksende yapılır: **sınıflandırma performansı** ve **SHAP feature importance**.

---

## İçindekiler

- [Motivasyon](#motivasyon)
- [Modeller](#modeller)
- [Deneysel Senaryolar](#deneysel-senaryolar)
- [Veri Seti](#veri-seti)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Sonuçlar](#sonuçlar)
- [Referanslar](#referanslar)

---

## Motivasyon

Geleneksel IDS sistemleri her saldırı tipini birbirinden bağımsız öğrenir. Gerçek dünyada ise saldırılar zaman içinde farklı kategorilerde gelir — yeni saldırı türleri eklenirken eski bilgi korunmalıdır. Bu *catastrophic forgetting* problemi, continual learning literatüründe iyi bilinen bir sorundur.

Bu çalışmanın katkıları:

1. **GNN tabanlı IDS** — ağ akışlarını node, benzerliklerini edge olarak modelleyerek akışlar arası ilişkiyi öğrenir.
2. **EWC ile continual learning** — Task A saldırıları öğrenildikten sonra Task B'ye geçişte eski bilgi EWC penalty ile korunur.
3. **E-GraphSAGE** — edge feature'larını aggregation'a dahil eden inductive GNN mimarisi; görülmemiş akışlara genelleyebilir.
4. **SHAP karşılaştırması** — üç modelin feature importance kararlarının nasıl farklılaştığını ortaya koyar.

---

## Modeller

### 1. GCN (Baseline)
**Dosya:** `models/elgnn.py`

Spektral graph convolution kullanır. GCNConv katmanları, her node'un komşularından ağırlıklı ortalama alarak yeni temsil öğrenir. EWC olmadan sadece Task B verisi üzerinde eğitilir — üst sınır ve alt sınır referansı olarak kullanılır.

```
Input (N, F)
    ↓  GCNConv (F → 128) + ReLU + Dropout
    ↓  GCNConv (128 → 64) + ReLU + Dropout
    ↓  Linear (64 → 32) + ReLU + Dropout
    ↓  Linear (32 → 2)
Output logits (N, 2)
```

### 2. GCN + EWC
**Dosya:** `models/elgnn.py` + `ewc.py`

GCN ile aynı mimari. Fark eğitim sürecinde: Task A tamamlandıktan sonra Fisher information matrix hesaplanır. Task B eğitiminde loss fonksiyonuna EWC penalty eklenir:

```
L_total = L_CE(Task B) + λ × Σ F_i × (θ_i - θ*_i)²
```

`λ = 100` (config.EWC_LAMBDA). Bu penalty, Task A için kritik ağırlıkların fazla değişmesini engeller.

### 3. E-GraphSAGE
**Dosya:** `models/e_graphsage.py`

Lo et al. (2022)'nin önerdiği edge-feature destekli GraphSAGE mimarisi. GCN'den iki temel farkı vardır:

**İnductive öğrenme:** SAGEConv, eğitimde görülmemiş node'lara genelleyebilir. GCNConv transductive'dir — tüm graph'ı eğitimde görmesi gerekir.

**Edge feature entegrasyonu:** Her mesaj geçişinde kaynak node feature'ı ile edge feature'ı birleştirilir:

```
edge_attr[u→v] = |x_u - x_v|           # mutlak fark, otomatik hesaplanır
message(u→v)   = ReLU(W_msg · [h_u || edge_attr])
aggregate(v)   = mean({message(u→v) for u ∈ N(v)})
update(v)      = W_out · [W_self · h_v || aggregate(v)]
```

Bu sayede sadece "komşu var mı" değil, "komşu ne kadar farklı" bilgisi de öğrenilir.

---

## Deneysel Senaryolar

Üç model üç farklı eğitim senaryosunda değerlendirilir:

| Senaryo | Model | Task A | Task B | EWC |
|---|---|---|---|---|
| A | GCN | ✗ | ✓ | ✗ |
| B | GCN + EWC | ✓ | ✓ | ✓ |
| C | E-GraphSAGE + EWC | ✓ | ✓ | ✓ |

**Task A saldırıları:** Reconnaissance, Fuzzers, Worms, Analysis (`data1`)

**Task B saldırıları:** Backdoor, DoS, Exploits, Generic, Shellcode (`data2`)

### Değerlendirme Metrikleri

Her model hem `data1/test` hem `data2/test` üzerinde değerlendirilir:

- Accuracy, Precision, Recall, F1-score (binary)
- ROC-AUC
- Confusion matrix
- **Backward Transfer (BWT):** Task B öğrendikten sonra Task A performansındaki düşüş
- **Forward Transfer (FWT):** Task A bilgisinin Task B öğrenimi hızına katkısı

### SHAP Karşılaştırması

Her model için `xai/selector.py` → `explain_shap.py` pipeline'ı çalışır:

- Her saldırı türü için top-10 feature
- Modeller arası feature önem sıralaması değişiyor mu?
- EWC'nin SHAP dağılımına etkisi var mı?
- E-GraphSAGE edge feature'ları node feature önemini nasıl etkiliyor?

---

## Veri Seti

**UNSW-NB15 / CIC-IDS** — CICFlowMeter ile işlenmiş ağ akışı verisi.

### Preprocessing Adımları

```
Ham CSV (CICFlowMeter_out.csv)
    ↓  splitter.py      data1 / data2 ayrımı (300K benign, orantılı dağıtım)
    ↓  cleaner.py       NaN, Inf, duplicate temizleme
    ↓  transforms.py    log1p → MinMaxScaler [0,1] (fit only on train)
    ↓  sampler.py       SMOTE + ENN (sadece train set)
    ↓
data1/ ve data2/
    ├── train.csv               (SMOTE+ENN sonrası, dengeli)
    ├── test.csv                (binary label)
    └── test_w_all_classes.csv  (multi-class label — SHAP için)
```

### Feature Mühendisliği

80+ feature'dan gereksizler drop edilir (`config/features.py`):

| Kategori | Sayı | Gerekçe |
|---|---|---|
| Identifier | 6 | IP/Port leakage riski |
| Zero variance | 9 | std = 0, bilgi yok |
| Dominant value | 2 | %99+ tek değer |
| Exact duplicate | 2 | Pearson = 1.0 |
| High correlation | 20+ | Pearson > 0.95, her iki dataset'te |

### Graph İnşası

Her node bir ağ akışı, k-NN (k=10) ile birbirine bağlanır. Öklid mesafesi üzerinden en yakın 10 komşu belirlenir → `edge_index` oluşturulur. E-GraphSAGE için `edge_attr = |x_src - x_dst|` otomatik hesaplanır.

---

## Proje Yapısı

```
elgnn-nids/
│
├── src/                    # Ana kaynak kodu paketi
│   ├── models/             # GNN implementasyonları
│   │   ├── __init__.py     # Model ve yardımcı fonksiyonları export
│   │   ├── base.py         # BaseGNN soyut sınıfı
│   │   ├── elgnn.py        # GCN (Senaryo A & B)
│   │   └── e_graphsage.py  # E-GraphSAGE (Senaryo C)
│   │
│   ├── preprocessing/       # Veri ön işleme pipeline'ı
│   │   ├── __init__.py     # Preprocessing fonksiyonları export
│   │   ├── cleaner.py      # NaN/dup temizleme, encoding
│   │   ├── io.py           # Ham CSV okuma / yazma
│   │   ├── pipeline.py     # Uçtan-uca pipeline + CLI
│   │   ├── sampler.py      # SMOTE + ENN
│   │   ├── splitter.py     # data1 / data2 oluşturma
│   │   └── transforms.py   # log1p, MinMaxScaler, apply_drops
│   │
│   ├── data/               # Veri yükleme ve şema tanımları
│   │   ├── __init__.py     # Veri yükleme fonksiyonları export
│   │   ├── schema.py       # TrainSplit, EvalSplit dataclass'ları
│   │   └── loader.py       # load_dataset, load_all_datasets, sanity_check
│   │
│   ├── xai/                # Explainable AI bileşenleri
│   │   ├── __init__.py     # XAI fonksiyonları export
│   │   ├── schema.py       # SampleSelection dataclass
│   │   └── selector.py     # SHAP sample seçimi + CLI
│   │
│   ├── utils/              # Yardımcı fonksiyonlar
│   │   ├── __init__.py     # Utility fonksiyonları export
│   │   └── logging.py      # banner(), step() context manager
│   │
│   └── config/             # Konfigürasyon dosyaları
│       ├── __init__.py     # Konfigürasyon modülleri export
│       ├── features.py     # Drop listeleri + gerekçeler
│       ├── hyperparams.py  # Model + eğitim + EWC sabitleri
│       └── paths.py        # Dosya yolları ve CSV isimleri
│
├── __init__.py             # Ana paket re-export'ları (geriye dönük uyumlu)
├── README.md               # Proje dokümantasyonu
├── ARCHITECTURE.md          # Mimari detayları
└── mnt/                    # Kullanıcı veri dizini
```

### Import Örnekleri

Yeni src layout ile import'lar:

```python
# Modeller
from src.models import ELGNN, EGraphSAGE, BaseGNN

# Veri yükleme
from src.data import load_dataset, sanity_check

# Ön işleme
from src.preprocessing import pipeline_main, clean

# XAI
from src.xai import selector_main

# Konfigürasyon
from src.config import hyperparams, features, paths

# Ana paket üzerinden (geriye dönük uyumlu)
import elgnn_nids
model = elgnn_nids.ELGNN(input_dim=44)
```
```

---

## 🚀 Hızlı Başlangıç

### **1. Kurulum**
```bash
# Ortam oluştur
conda create -n gnn-ids python=3.10
conda activate gnn-ids

# Bağımlılıkları kur
pip install -r requirements.txt
```

### **2. Test Etme (Orijinal Veri ile)**
```bash
# DoS saldırısı test et (1000 örnek)
python tests/test_original_simple.py --data_path /path/to/CICFlowMeter_out.csv --attack DoS --samples 1000

# PortScan saldırısı test et (E-GraphSAGE model)
python tests/test_original_simple.py --attack PortScan --model_type egraphsage --samples 2000

# Birden fazla saldırı test et
for attack in DoS PortScan Bot; do
    python tests/test_original_simple.py --attack $attack --samples 1000
done
```

### **3. Model Eğitimi**
```bash
# Senaryo A: GCN baseline (Sadece Task B)
python train.py --scenario gcn_baseline

# Senaryo B: GCN + EWC (Task A → Task B)
python train.py --scenario gcn_ewc

# Senaryo C: E-GraphSAGE + EWC (Task A → Task B)
python train.py --scenario egraphsage_ewc
```

### **4. Eğitilmiş Modelleri Test Etme**
```bash
# Eğitilmiş modeli test et
python evaluate.py --model_path outputs/gcn_ewc_model.pt --dataset both --model_type gcn

# SHAP analizi yap
python explain_shap.py --model_path outputs/gcn_ewc_model.pt --dataset data2 --model_type gcn
```

## 📁 Proje Yapısı

```
elgnn-nids/
│
├── src/                    # Ana kaynak kodu paketi
│   ├── models/             # GNN implementasyonları
│   ├── preprocessing/       # Veri ön işleme
│   ├── data/               # Veri yükleme
│   ├── xai/                # Explainable AI
│   ├── utils/              # Yardımcı fonksiyonlar
│   └── config/             # Konfigürasyon
│
├── tests/                   # Test scriptleri
│   ├── test_original_simple.py    # ⭐ Orijinal veri testi (önerilen)
│   ├── test_original_data_sampled.py  # Gelişmiş test
│   └── test_models.py              # İşlenmiş veri testi
│
├── requirements.txt          # Python bağımlılıkları
├── .gitignore             # Git için dosya hariç tutma
├── README.md              # Proje dokümantasyonu
└── TRAINING_GUIDE.md      # Detaylı eğitim rehberi
```

## 🎯 Test Komutları

### **Orijinal Veri ile Test Etme**
```bash
# Temel test (önerilen)
python tests/test_original_simple.py --data_path /Users/cansu/Downloads/CICFlowMeter_out.csv --attack DoS --samples 1000

# Tüm saldırı türlerini test et
python tests/test_original_simple.py --data_path /path/to/CICFlowMeter_out.csv --attack all --samples 5000
```

### **Model Eğitimi**
```bash
# CPU ile eğitim
python train.py --scenario gcn_baseline --device cpu

# GPU ile eğitim (varsa)
python train.py --scenario gcn_ewc --device cuda
```

## 📊 Beklenen Çıktılar

### **Test Sonuçları**
- **Accuracy**: Sınıflandırma doğruluğu
- **Detection Rate**: True Positive oranı
- **False Alarm Rate**: False Positive oranı  
- **Miss Rate**: False Negative oranı

### **Model Karşılaştırma**
| Model | Task A | Task B | BWT |
|--------|----------|----------|-----|
| GCN | ~0.85 | ~0.88 | -0.15 |
| GCN+EWC | ~0.82 | ~0.86 | -0.05 |
| E-GraphSAGE+EWC | ~0.84 | ~0.89 | -0.03 |

---

## Kullanım

### 1. Preprocessing

```bash
python -m src.preprocessing.pipeline
```

Ham CSV'den `not_stratified_no_mi/data1/` ve `data2/` oluşturulur.

### 2. Veri Kontrolü

```python
from src.data import sanity_check
sanity_check(verbose=True)
# Tüm split'lerin aynı feature setinde olduğunu doğrular
```

### 3. Model İnstansiasyonu

```python
from src.models import ELGNN, EGraphSAGE

# GCN (Scenario A ve B için aynı mimari)
gcn = ELGNN(input_dim=44)

# E-GraphSAGE (Scenario C)
sage = EGraphSAGE(input_dim=44)

# Forward pass (aynı arayüz, üç model için de çalışır)
logits = model(x, edge_index)              # GCN
logits = model(x, edge_index)              # E-GraphSAGE (edge_attr otomatik)
logits = model(x, edge_index, edge_attr)   # E-GraphSAGE (explicit edge_attr)
```

### 4. Eğitim

```bash
# Scenario A: GCN baseline
python train.py --scenario gcn_baseline

# Scenario B: GCN + EWC
python train.py --scenario gcn_ewc

# Scenario C: E-GraphSAGE + EWC
python train.py --scenario e_graphsage_ewc
```

### 5. SHAP Analizi

```bash
# Sample seçimi
python -m src.xai.selector

# Açıklama
python explain_shap.py
```

---

## Sonuçlar

*(Eğitim tamamlandıktan sonra doldurulacak)*

### Performans Karşılaştırması

| Model | data1 F1 | data2 F1 | AUC | BWT |
|---|---|---|---|---|
| GCN (baseline) | — | — | — | — |
| GCN + EWC | — | — | — | — |
| E-GraphSAGE + EWC | — | — | — | — |

### SHAP Feature Önem Karşılaştırması

*(Grafik ve tablo buraya eklenecek)*

---

## Referanslar

- **EWC:** Kirkpatrick et al. (2017) *Overcoming catastrophic forgetting in neural networks*. PNAS.
- **GCN:** Kipf & Welling (2017) *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR.
- **GraphSAGE:** Hamilton et al. (2017) *Inductive Representation Learning on Large Graphs*. NeurIPS.
- **E-GraphSAGE:** Lo et al. (2022) *E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT*. NDSS.
- **SHAP:** Lundberg & Lee (2017) *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
