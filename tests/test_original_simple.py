"""
test_original_simple.py

Simple testing with original CICFlowMeter_out.csv data.
Uses a straightforward approach without complex graph construction.

Usage:
    python test_original_simple.py --data_path /Users/cansu/Downloads/CICFlowMeter_out.csv --attack DoS --samples 1000
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from src.models import ELGNN, EGraphSAGE
from src.config import hyperparams as config


def load_and_sample_original_data(data_path: str, attack_type: str, 
                               total_samples: int = 1000) -> pd.DataFrame:
    """Load and sample original data efficiently."""
    print(f"📁 Loading original data from {data_path}")
    
    try:
        # Load data in chunks to avoid memory issues
        df_chunks = []
        chunk_size = 50000
        for chunk in pd.read_csv(data_path, chunksize=chunk_size):
            df_chunks.append(chunk)
            if len(df_chunks) * chunk_size > total_samples * 2:  # Stop early
                break
        
        df = pd.concat(df_chunks, ignore_index=True)
        print(f"✅ Loaded {len(df):,} samples from first chunks")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Get unique attack types
    if 'Label' not in df.columns:
        print("❌ 'Label' column not found")
        return None
    
    unique_attacks = df['Label'].unique()
    print(f"📊 Available attacks: {list(unique_attacks)}")
    
    # Filter for selected attack + Benign
    selected_attacks = [attack_type, 'Benign']
    filtered_df = df[df['Label'].isin(selected_attacks)]
    
    print(f"🎯 Filtered to {len(filtered_df):,} samples for {attack_type}")
    print(f"📈 Attack distribution:")
    for attack in selected_attacks:
        count = len(filtered_df[filtered_df['Label'] == attack])
        print(f"   {attack:15s}: {count:6,} samples")
    
    # Stratified sampling
    if len(filtered_df) > total_samples:
        attack_samples = filtered_df[filtered_df['Label'] == attack_type]
        benign_samples = filtered_df[filtered_df['Label'] == 'Benign']
        
        # Calculate sample sizes (70% attack, 30% benign)
        n_attack = min(len(attack_samples), int(total_samples * 0.7))
        n_benign = min(len(benign_samples), total_samples - n_attack)
        
        # Sample from each group
        if len(attack_samples) > n_attack:
            attack_sampled = attack_samples.sample(n=n_attack, random_state=42)
        else:
            attack_sampled = attack_samples
        
        if len(benign_samples) > n_benign:
            benign_sampled = benign_samples.sample(n=n_benign, random_state=42)
        else:
            benign_sampled = benign_samples
        
        # Combine and shuffle
        sampled_df = pd.concat([attack_sampled, benign_sampled], ignore_index=True)
        sampled_df = sampled_df.sample(frac=1.0, random_state=42)
        
        print(f"📈 Sampled: {len(sampled_df)} total ({n_attack} attack, {n_benign} benign)")
    else:
        sampled_df = filtered_df
        print(f"📈 Using all {len(sampled_df)} samples")
    
    return sampled_df


def preprocess_simple(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Simple preprocessing for testing."""
    print("🔄 Preprocessing data...")
    
    # Extract labels
    if 'Label' in df.columns:
        labels = df['Label'].values
        feature_df = df.drop(columns=['Label'])
    else:
        labels = np.zeros(len(df))
        feature_df = df
    
    # Convert to binary labels
    binary_labels = np.array([0 if label == 'Benign' else 1 for label in labels])
    
    # Handle missing values
    feature_df = feature_df.fillna(0)
    
    # Remove non-numeric columns
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    feature_df = feature_df[numeric_cols]
    print(f"📊 Using {len(numeric_cols)} numeric features")
    
    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(feature_df.values)
    
    # Convert to tensors
    X = torch.tensor(scaled_features, dtype=torch.float32)
    y = torch.tensor(binary_labels, dtype=torch.long)
    
    print(f"✅ Preprocessed: {X.shape}")
    print(f"📊 Label distribution: {np.bincount(binary_labels)}")
    
    return X, y


def create_sequential_graph(X: torch.Tensor) -> torch.Tensor:
    """Create simple sequential graph for testing."""
    print("🔗 Creating sequential graph...")
    
    N = X.shape[0]
    
    # Create simple sequential connections
    edge_list = []
    for i in range(N - 1):
        edge_list.append([i, i + 1])
        edge_list.append([i + 1, i])  # Bidirectional
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    print(f"✅ Sequential graph: {N} nodes, {edge_index.shape[1]} edges")
    return edge_index


def test_model_simple(X: torch.Tensor, y: torch.Tensor, 
                    model_type: str = "gcn") -> dict:
    """Test model with simple approach."""
    print(f"\n🧪 Testing {model_type.upper()} model...")
    
    # Create simple graph
    edge_index = create_sequential_graph(X)
    
    # Initialize model
    if model_type == "gcn":
        model = ELGNN(
            input_dim=X.shape[1],
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            num_classes=config.NUM_CLASSES
        )
    elif model_type == "egraphsage":
        model = EGraphSAGE(
            input_dim=X.shape[1],
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            num_classes=config.NUM_CLASSES
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(X, edge_index)
        _, predicted = torch.max(outputs, 1)
    
    # Calculate metrics
    y_true = y.numpy()
    y_pred = predicted.numpy()
    
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"✅ {model_type.upper()} Accuracy: {accuracy:.4f}")
    
    # Detailed metrics
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    
    print(f"📊 Confusion Matrix:")
    print(f"   True Negatives:  {tn:6d} ({tn/total:6.2%})")
    print(f"   False Positives: {fp:6d} ({fp/total:6.2%})")
    print(f"   False Negatives: {fn:6d} ({fn/total:6.2%})")
    print(f"   True Positives:  {tp:6d} ({tp/total:6.2%})")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Attack']))
    
    return {
        'model_type': model_type,
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'num_samples': len(y_true),
        'num_features': X.shape[1],
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Simple testing with original data')
    parser.add_argument('--data_path', type=str, 
                       default='/Users/cansu/Downloads/CICFlowMeter_out.csv',
                       help='Path to CICFlowMeter_out.csv')
    parser.add_argument('--attack', type=str, default='DoS',
                       help='Attack type to test')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Total samples to test')
    parser.add_argument('--model_type', type=str, default='gcn',
                       choices=['gcn', 'egraphsage'],
                       help='Model type')
    
    args = parser.parse_args()
    
    print("🚀 ELGNN-NIDS Simple Original Data Testing")
    print("=" * 60)
    
    # Load and sample data
    df = load_and_sample_original_data(args.data_path, args.attack, args.samples)
    if df is None:
        return
    
    # Preprocess
    X, y = preprocess_simple(df)
    
    # Test model
    results = test_model_simple(X, y, args.model_type)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 60)
    print(f"Attack Type: {args.attack}")
    print(f"Model: {args.model_type.upper()}")
    print(f"Total Samples: {results['num_samples']:,}")
    print(f"Features: {results['num_features']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    
    # Performance metrics
    total = results['num_samples']
    print(f"\n📈 Performance Metrics:")
    print(f"Detection Rate: {(results['true_positives']/total)*100:.2f}%")
    print(f"False Alarm Rate: {(results['false_positives']/total)*100:.2f}%")
    print(f"Miss Rate: {(results['false_negatives']/total)*100:.2f}%")
    
    print("=" * 60)
    print("✅ Testing completed successfully!")
    
    print("\n💡 Next steps:")
    print("1. Train models: python train.py --scenario gcn_baseline")
    print("2. Test other attacks: python test_original_simple.py --attack PortScan")
    print("3. Try E-GraphSAGE: python test_original_simple.py --model_type egraphsage")


if __name__ == "__main__":
    main()
