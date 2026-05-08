# ELGNN-NIDS Testing Scripts

This directory contains testing scripts for ELGNN-NIDS models with original CICFlowMeter data.

## 📁 Test Scripts Overview

### **test_original_simple.py** ⭐ **Recommended**
Efficient testing with original CICFlowMeter data.
- Loads data in chunks to avoid memory issues
- Samples attack types with stratified distribution
- Uses sequential graph construction for stability
- Provides comprehensive evaluation metrics

**Usage:**
```bash
python tests/test_original_simple.py --data_path /path/to/CICFlowMeter_out.csv --attack DoS --samples 1000
```

**Available Attacks:**
- DoS, DDoS, PortScan, Bot, Exploits, Fuzzers, Worms, Backdoor, Analysis, Generic, Shellcode, Reconnaissance

### **test_original_data_sampled.py**
Advanced testing with k-NN graph construction.
- Uses sklearn for efficient neighbor finding
- Supports larger sample sizes
- More complex graph construction

### **test_original_data.py**
Original version (deprecated - use test_original_simple.py instead).

### **test_models.py**
Testing with pre-built graphs from processed data.

## 🎯 Quick Testing Commands

```bash
# Test DoS attack with 1000 samples
python tests/test_original_simple.py --attack DoS --samples 1000

# Test PortScan attack with E-GraphSAGE
python tests/test_original_simple.py --attack PortScan --model_type egraphsage --samples 2000

# Test multiple attacks
for attack in DoS PortScan Bot; do
    python tests/test_original_simple.py --attack $attack --samples 1000
done
```

## 📊 Expected Output

Each test provides:
- **Accuracy**: Overall classification accuracy
- **Detection Rate**: True Positive Rate
- **False Alarm Rate**: False Positive Rate  
- **Miss Rate**: False Negative Rate
- **Confusion Matrix**: Detailed classification results
- **Classification Report**: Precision, Recall, F1-score

## 🔧 Requirements

Make sure you have the required dependencies:
```bash
pip install torch torch-geometric pandas numpy scikit-learn
```

## 📁 Data Requirements

The test scripts expect:
- **CICFlowMeter_out.csv** with 'Label' column
- **Attack types** in ['Benign', 'DoS', 'DDoS', 'PortScan', 'Bot', 'Exploits', 'Fuzzers', 'Worms', 'Backdoor', 'Analysis', 'Generic', 'Shellcode', 'Reconnaissance']

## 🚨 Troubleshooting

**Memory Issues**: Use smaller `--samples` parameter
**Column Errors**: Scripts automatically handle non-numeric columns
**Graph Issues**: Sequential graph construction is most stable
**Performance**: Untrained models will have low accuracy (~10-20%)
