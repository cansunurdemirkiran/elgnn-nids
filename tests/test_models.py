"""
test_models.py

Test ELGNN-NIDS models with existing data1 and data2 graphs.
This script demonstrates model loading, inference, and basic evaluation without training.

Usage:
    python test_models.py
"""

from __future__ import annotations

import time
from typing import Dict

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from graph_builder import load_graph
from src.models import ELGNN, EGraphSAGE
from src.config import hyperparams as config


def test_model_inference(model_type: str = "gcn", dataset_name: str = "data2") -> Dict:
    """
    Test model inference on specified dataset.
    
    Args:
        model_type: "gcn" or "egraphsage"
        dataset_name: "data1" or "data2"
        
    Returns:
        Dictionary with test results
    """
    print(f"\n🧪 Testing {model_type.upper()} model on {dataset_name.upper()}...")
    
    device = torch.device('cpu')  # Use CPU for testing
    
    # Load data and graph
    try:
        x, y, edge_index, edge_attr = load_graph(dataset_name, 'test')
        print(f"✅ Loaded {x.shape[0]} samples with {x.shape[1]} features")
    except FileNotFoundError:
        print(f"❌ Graph files not found for {dataset_name}/test")
        return {}
    
    # Move to device
    x, y = x.to(device), y.to(device)
    edge_index = edge_index.to(device)
    
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)
    
    # Initialize model
    if model_type == "gcn":
        model = ELGNN(
            input_dim=x.shape[1],
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            num_classes=config.NUM_CLASSES
        )
    elif model_type == "egraphsage":
        model = EGraphSAGE(
            input_dim=x.shape[1],
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            num_classes=config.NUM_CLASSES
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    model.eval()
    
    # Test forward pass
    start_time = time.time()
    
    with torch.no_grad():
        if edge_attr is not None:
            outputs = model(x, edge_index, edge_attr)
        else:
            outputs = model(x, edge_index)
        
        # Get predictions
        _, predicted = torch.max(outputs, 1)
        
        # Get probabilities
        probs = torch.softmax(outputs, dim=1)
    
    inference_time = time.time() - start_time
    
    # Calculate metrics
    y_true = y.cpu().numpy()
    y_pred = predicted.cpu().numpy()
    y_proba = probs.cpu().numpy()
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"✅ Inference completed in {inference_time:.3f} seconds")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ True Positives: {tp}, True Negatives: {tn}")
    print(f"✅ False Positives: {fp}, False Negatives: {fn}")
    
    return {
        'model_type': model_type,
        'dataset': dataset_name,
        'accuracy': accuracy,
        'inference_time': inference_time,
        'num_samples': len(y_true),
        'num_features': x.shape[1],
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'confusion_matrix': cm.tolist()
    }


def test_all_combinations():
    """Test all model-dataset combinations."""
    print("🚀 Testing all model-dataset combinations...")
    
    results = {}
    
    # Test combinations
    combinations = [
        ("gcn", "data1"),
        ("gcn", "data2"),
        ("egraphsage", "data1"),
        ("egraphsage", "data2"),
    ]
    
    for model_type, dataset_name in combinations:
        try:
            result = test_model_inference(model_type, dataset_name)
            if result:
                key = f"{model_type}_{dataset_name}"
                results[key] = result
        except Exception as e:
            print(f"❌ Error testing {model_type} on {dataset_name}: {e}")
    
    return results


def compare_results(results: Dict):
    """Compare and display results."""
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON RESULTS")
    print("="*60)
    
    if not results:
        print("❌ No results to compare")
        return
    
    # Create comparison table
    print(f"{'Model':12s} {'Dataset':10s} {'Accuracy':10s} {'Time(s)':8s} {'Samples':8s}")
    print("-" * 60)
    
    for key, result in results.items():
        print(f"{result['model_type']:12s} {result['dataset']:10s} "
              f"{result['accuracy']:10.4f} {result['inference_time']:8.3f} "
              f"{result['num_samples']:8d}")
    
    print("-" * 60)
    
    # Find best performing model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best performing model: {best_model[0]}")
    print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
    print(f"   Dataset: {best_model[1]['dataset']}")
    
    # Find fastest model
    fastest_model = min(results.items(), key=lambda x: x[1]['inference_time'])
    print(f"\n⚡ Fastest model: {fastest_model[0]}")
    print(f"   Inference time: {fastest_model[1]['inference_time']:.3f}s")
    print(f"   Dataset: {fastest_model[1]['dataset']}")


def test_model_parameters():
    """Test model parameter counts and memory usage."""
    print("\n🔧 Testing model parameters...")
    
    input_dim = 44  # Default feature dimension
    
    models = {
        "GCN": ELGNN(input_dim=input_dim),
        "E-GraphSAGE": EGraphSAGE(input_dim=input_dim)
    }
    
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n{name}:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Estimate memory usage (rough approximation)
        param_size = total_params * 4  # 4 bytes per float32
        print(f"  Estimated memory: {param_size / 1024 / 1024:.2f} MB")


def main():
    """Main testing function."""
    print("🧪 ELGNN-NIDS Model Testing Suite")
    print("=" * 50)
    
    # Test model parameters
    test_model_parameters()
    
    # Test all combinations
    results = test_all_combinations()
    
    # Compare results
    compare_results(results)
    
    print("\n✅ Testing completed!")
    print("\n💡 Next steps:")
    print("   1. Train models: python train.py --scenario gcn_baseline")
    print("   2. Evaluate trained models: python evaluate.py --model_path outputs/model.pt")
    print("   3. Run SHAP analysis: python explain_shap.py")


if __name__ == "__main__":
    main()
