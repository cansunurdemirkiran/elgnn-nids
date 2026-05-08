# ELGNN-NIDS Training and Testing Guide

This comprehensive guide covers how to train and test the ELGNN-NIDS models using the new src layout structure.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Preparation](#data-preparation)
3. [Graph Construction](#graph-construction)
4. [Model Training](#model-training)
5. [Model Evaluation](#model-evaluation)
6. [SHAP Analysis](#shap-analysis)
7. [Troubleshooting](#troubleshooting)
8. [Expected Results](#expected-results)

## 🚀 Prerequisites

### Environment Setup

```bash
# Create conda environment
conda create -n gnn-ids python=3.10
conda activate gnn-ids

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric

# Install other dependencies
pip install pandas numpy scikit-learn imbalanced-learn shap tqdm matplotlib seaborn
```

### Required Files

- **Raw Data**: `CICFlowMeter_out.csv` (network traffic data)
- **Project Structure**: Ensure all files are in the new src layout

## 📊 Data Preparation

### Step 1: Run Preprocessing Pipeline

```bash
cd /Users/cansu/Desktop/Master/Deep\ Learning/project/elgnn-nids

# Run preprocessing pipeline
python -m src.preprocessing.pipeline
```

**What this does:**
- Cleans raw data (removes NaN, duplicates, encodes categoricals)
- Splits into data1 (Task A) and data2 (Task B) datasets
- Applies feature transformations (log1p, MinMax scaling)
- Applies SMOTE+ENN sampling to training sets
- Creates processed datasets in `not_stratified_no_mi/data1/` and `not_stratified_no_mi/data2/`

**Expected Output:**
```
not_stratified_no_mi/
├── data1/
│   ├── train.csv
│   ├── test.csv
│   └── test_w_all_classes.csv
└── data2/
    ├── train.csv
    ├── test.csv
    └── test_w_all_classes.csv
```

### Step 2: Verify Data Integrity

```python
# Create verification script
python -c "
from src.data import sanity_check
sanity_check(verbose=True)
print('✅ Data verification completed successfully!')
"
```

## 🔗 Graph Construction

### Step 3: Build k-NN Graphs

```bash
# Build graphs for all datasets
python graph_builder.py
```

**What this does:**
- Constructs k-NN graphs (k=10) for each dataset split
- Computes edge features as absolute differences between node features
- Saves graphs to `graphs/` directory
- Creates undirected graphs with bidirectional edges

**Expected Output:**
```
graphs/
├── data1_train_graph.pt
├── data1_test_graph.pt
├── data2_train_graph.pt
└── data2_test_graph.pt
```

### Step 4: Test Graph Loading

```python
# Test graph loading
python -c "
from graph_builder import load_graph
x, y, edge_index, edge_attr = load_graph('data1', 'train')
print(f'✅ Graph loaded: {x.shape[0]} nodes, {edge_index.shape[1]} edges')
"
```

## 🏋️ Model Training

### Training Scenarios

The project supports three training scenarios:

1. **Scenario A**: GCN baseline (Task B only)
2. **Scenario B**: GCN + EWC (Task A → Task B with continual learning)
3. **Scenario C**: E-GraphSAGE + EWC (Task A → Task B with continual learning)

### Scenario A: GCN Baseline

```bash
# Train GCN baseline on Task B only
python train.py --scenario gcn_baseline --device cuda
```

**What this does:**
- Trains GCN model only on Task B data (data2)
- Evaluates on both Task A and Task B test sets
- Measures backward transfer (performance on Task A without training on it)

**Expected Output:**
```
outputs/gcn_baseline_results.txt
```

### Scenario B: GCN + EWC

```bash
# Train GCN with EWC continual learning
python train.py --scenario gcn_ewc --device cuda
```

**What this does:**
- Trains GCN on Task A (data1)
- Computes Fisher information matrix
- Trains on Task B (data2) with EWC penalty
- Measures catastrophic forgetting and backward transfer

**Expected Output:**
```
outputs/gcn_ewc_results.txt
```

### Scenario C: E-GraphSAGE + EWC

```bash
# Train E-GraphSAGE with EWC continual learning
python train.py --scenario egraphsage_ewc --device cuda
```

**What this does:**
- Trains E-GraphSAGE on Task A (data1)
- Computes Fisher information matrix
- Trains on Task B (data2) with EWC penalty and edge features
- Measures performance improvement from edge features

**Expected Output:**
```
outputs/egraphsage_ewc_results.txt
```

### Training Parameters

Key training parameters (configurable in `src/config/hyperparams.py`):

```python
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_DIM = 128
OUTPUT_DIM = 64
NUM_CLASSES = 2
EWC_LAMBDA = 100
WEIGHT_DECAY = 1e-5
```

## 📈 Model Evaluation

### Evaluation Metrics

Each training scenario evaluates:

1. **Task A Performance**: Accuracy on Task A test set
2. **Task B Performance**: Accuracy on Task B test set
3. **Backward Transfer (BWT)**: Change in Task A performance after Task B training

### Detailed Evaluation Script

```python
# Create detailed evaluation script
python -c "
import torch
from graph_builder import load_graph
from src.models import ELGNN, EGraphSAGE
from sklearn.metrics import classification_report, confusion_matrix

# Load model and data
model = ELGNN(input_dim=44)
model.load_state_dict(torch.load('outputs/gcn_ewc_model.pt'))
model.eval()

x, y, edge_index, edge_attr = load_graph('data2', 'test')

# Get predictions
with torch.no_grad():
    outputs = model(x, edge_index)
    _, predicted = torch.max(outputs, 1)

# Print detailed metrics
print(classification_report(y.numpy(), predicted.numpy()))
print('Confusion Matrix:')
print(confusion_matrix(y.numpy(), predicted.numpy()))
"
```

## 🔍 SHAP Analysis

### Step 5: Sample Selection

```bash
# Select samples for SHAP analysis
python -m src.xai.selector
```

**What this does:**
- Loads trained model (Task B only)
- Selects correctly predicted samples from each attack type
- Saves sample selections to `xai_outputs/`

### Step 6: SHAP Explanation

```bash
# Explain GCN model on data2
python explain_shap.py --model_path outputs/gcn_ewc_model.pt --dataset data2 --model_type gcn

# Explain E-GraphSAGE model on data1
python explain_shap.py --model_path outputs/egraphsage_ewc_model.pt --dataset data1 --model_type egraphsage
```

**What this does:**
- Computes SHAP values for selected samples
- Generates feature importance plots
- Creates SHAP summary visualizations
- Saves results to `xai_outputs/`

**Expected Output:**
```
xai_outputs/
├── data2/
│   ├── selected_samples.json
│   ├── gcn_feature_importance.png
│   ├── gcn_shap_summary.png
│   └── gcn_shap_results.json
└── data1/
    ├── selected_samples.json
    ├── egraphsage_feature_importance.png
    ├── egraphsage_shap_summary.png
    └── egraphsage_shap_results.json
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Use CPU or reduce batch size
python train.py --scenario gcn_ewc --device cpu
```

#### 2. Graph File Not Found

```bash
# Rebuild graphs
python graph_builder.py
```

#### 3. Data Loading Issues

```bash
# Re-run preprocessing
python -m src.preprocessing.pipeline
```

#### 4. Import Errors

```bash
# Test imports
python -c "
from src.models import ELGNN, EGraphSAGE
from src.data import load_dataset
from graph_builder import load_graph
print('✅ All imports successful!')
"
```

### Debug Mode

For debugging, you can run with reduced parameters:

```python
# Create debug script
import torch
from src.config import hyperparams as config

# Reduce parameters for debugging
config.NUM_EPOCHS = 5
config.HIDDEN_DIM = 32
config.LEARNING_RATE = 0.01

print("Debug parameters set!")
```

## 📊 Expected Results

### Performance Benchmarks

Based on the original paper, expect these approximate results:

| Model | Task A F1 | Task B F1 | BWT |
|-------|-----------|-----------|-----|
| GCN (baseline) | ~0.85 | ~0.88 | -0.15 |
| GCN + EWC | ~0.82 | ~0.86 | -0.05 |
| E-GraphSAGE + EWC | ~0.84 | ~0.89 | -0.03 |

### Feature Importance Analysis

- **Top features** typically include: packet sizes, flow duration, TCP flags
- **E-GraphSAGE** should show different feature importance patterns due to edge features
- **EWC models** should maintain more stable feature importance across tasks

### Output Files Structure

```
elgnn-nids/
├── graphs/                     # k-NN graphs
├── outputs/                    # Training results
│   ├── gcn_baseline_results.txt
│   ├── gcn_ewc_results.txt
│   └── egraphsage_ewc_results.txt
├── xai_outputs/               # SHAP analysis results
│   ├── data1/
│   └── data2/
└── not_stratified_no_mi/      # Processed datasets
    ├── data1/
    └── data2/
```

## 🎯 Quick Start Script

For a complete end-to-end run:

```bash
#!/bin/bash
# complete_training.sh

echo "🚀 Starting ELGNN-NIDS Training Pipeline"

# 1. Preprocessing
echo "📊 Running preprocessing..."
python -m src.preprocessing.pipeline

# 2. Graph Construction
echo "🔗 Building graphs..."
python graph_builder.py

# 3. Training all scenarios
echo "🏋️ Training models..."
python train.py --scenario gcn_baseline
python train.py --scenario gcn_ewc
python train.py --scenario egraphsage_ewc

# 4. SHAP Analysis
echo "🔍 Running SHAP analysis..."
python -m src.xai.selector
python explain_shap.py --model_path outputs/gcn_ewc_model.pt --dataset data2 --model_type gcn
python explain_shap.py --model_path outputs/egraphsage_ewc_model.pt --dataset data1 --model_type egraphsage

echo "✅ Training pipeline completed!"
echo "📊 Check outputs/ and xai_outputs/ for results"
```

Make it executable and run:
```bash
chmod +x complete_training.sh
./complete_training.sh
```

## 📝 Next Steps

After successful training and testing:

1. **Analyze Results**: Compare performance across scenarios
2. **Feature Analysis**: Examine SHAP feature importance
3. **Hyperparameter Tuning**: Optimize for better performance
4. **Extension**: Add new models or datasets
5. **Publication**: Prepare results for academic publication

This guide provides a complete workflow for training and testing the ELGNN-NIDS models with the new professional src layout structure.
