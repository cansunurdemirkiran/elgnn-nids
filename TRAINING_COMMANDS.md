# ELGNN-NIDS Model Training Commands

## 🚀 Quick Training Guide

### **Prerequisites**
```bash
# 1. Ensure graphs are built
python graph_builder.py

# 2. Check data availability
ls not_stratified_no_mi/
ls graphs/
```

## 📋 Training Scenarios

### **Scenario A: GCN Baseline (Task B Only)**
```bash
# CPU training
python train_models.py --scenario gcn_baseline --device cpu --epochs 50

# GPU training (if available)
python train_models.py --scenario gcn_baseline --device cuda --epochs 100
```

### **Scenario B: GCN + EWC (Task A → Task B)**
```bash
# Two-phase training with continual learning
python train_models.py --scenario gcn_ewc --device cpu --epochs 50

# Longer training for better performance
python train_models.py --scenario gcn_ewc --device cuda --epochs 100
```

### **Scenario C: E-GraphSAGE + EWC (Task A → Task B)**
```bash
# Most advanced scenario
python train_models.py --scenario egraphsage_ewc --device cpu --epochs 50

# For research-level training
python train_models.py --scenario egraphsage_ewc --device cuda --epochs 200
```

## 📊 Expected Outputs

### **Model Files**
- `outputs/gcn_baseline_model.pt` - GCN baseline model
- `outputs/gcn_ewc_model.pt` - GCN with EWC model  
- `outputs/egraphsage_ewc_model.pt` - E-GraphSAGE with EWC model

### **Training Logs**
```
🚀 ELGNN-NIDS Model Training
============================================================
Scenario: gcn_ewc
Device: cpu
Epochs: 50
============================================================
📚 Scenario B: GCN + EWC (Task A → Task B)

🔄 Phase 1: Training on Task A
✅ Loaded data1/train: 17077 samples, 38 features
✅ Loaded data1/test: 4499 samples, 38 features
Epoch   1: Train Loss: 0.6931, Val Loss: 0.6931, Val Acc: 0.5000
...
✅ Task A weights stored for EWC

🔄 Phase 2: Training on Task B with EWC
✅ Loaded data2/train: 19397 samples, 38 features
✅ Loaded data2/test: 5096 samples, 38 features
Epoch   1: Train Loss: 0.6931, Val Loss: 0.6931, Val Acc: 0.5000
...
✅ Training completed. Best Val Acc: 0.8566
💾 Model saved: outputs/gcn_ewc_model.pt
📊 Task A Acc: 0.8234, Task B Acc: 0.8566
============================================================
✅ gcn_ewc training completed!
```

## 🔧 Troubleshooting

### **Common Issues**

**Graph Files Not Found**
```bash
❌ Graph files not found for data1/train
💡 Run: python graph_builder.py
```

**CUDA Out of Memory**
```bash
# Use CPU instead
python train_models.py --scenario gcn_baseline --device cpu

# Reduce epochs
python train_models.py --scenario gcn_ewc --device cuda --epochs 20
```

**Data Loading Issues**
```bash
# Check preprocessing
python -m src.preprocessing.pipeline

# Verify data files
ls not_stratified_no_mi/data1/
ls not_stratified_no_mi/data2/
```

## 📈 Performance Expectations

### **Based on Research Paper**
| Scenario | Task A Acc | Task B Acc | BWT |
|----------|-------------|-------------|------|
| GCN Baseline | - | ~0.88 | - |
| GCN + EWC | ~0.82 | ~0.86 | -0.05 |
| E-GraphSAGE + EWC | ~0.84 | ~0.89 | -0.03 |

### **Training Time Estimates**
- **CPU**: 5-15 minutes per scenario
- **GPU**: 1-5 minutes per scenario
- **EWC scenarios**: 2x longer (two-phase training)

## 🎯 After Training

### **1. Evaluate Models**
```bash
# Comprehensive evaluation
python evaluate.py --model_path outputs/gcn_ewc_model.pt --dataset both --model_type gcn

# Quick testing
python tests/test_models.py
```

### **2. Test with Original Data**
```bash
# Test trained model on original CICFlowMeter data
python tests/test_original_simple.py --model_path outputs/gcn_ewc_model.pt --data_path /path/to/CICFlowMeter_out.csv --attack DoS
```

### **3. SHAP Analysis**
```bash
# Feature importance analysis
python explain_shap.py --model_path outputs/gcn_ewc_model.pt --dataset data2 --model_type gcn
```

## 🚀 Batch Training

### **Train All Scenarios**
```bash
#!/bin/bash
# train_all.sh

echo "Training all scenarios..."

echo "Scenario A: GCN Baseline"
python train_models.py --scenario gcn_baseline --device cpu --epochs 50

echo "Scenario B: GCN + EWC"
python train_models.py --scenario gcn_ewc --device cpu --epochs 50

echo "Scenario C: E-GraphSAGE + EWC"
python train_models.py --scenario egraphsage_ewc --device cpu --epochs 50

echo "All training completed!"
```

```bash
# Make executable and run
chmod +x train_all.sh
./train_all.sh
```

## 📝 Notes

- **EWC Lambda**: Set to 100 (config.EWC_LAMBDA)
- **Learning Rate**: 0.001 (config.LEARNING_RATE)
- **Early Stopping**: Patience = 10 epochs
- **Batch Size**: Full dataset (no batching for simplicity)
- **Model Architecture**: 2-layer GCN/GraphSAGE with ReLU
