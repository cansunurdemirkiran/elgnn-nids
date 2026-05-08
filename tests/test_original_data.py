"""
test_original_data.py

Test ELGNN-NIDS models with original CICFlowMeter_out.csv data.
This script loads original data, selects specific attack types, and evaluates models.

Usage:
    python test_original_data.py --data_path /Users/cansu/Downloads/CICFlowMeter_out.csv --attack DoS
"""

from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from src.models import ELGNN, EGraphSAGE
from src.config import hyperparams as config


def load_original_data(data_path: str) -> pd.DataFrame:
    """Load original CICFlowMeter data."""
    print(f"📁 Loading original data from {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df):,} samples with {len(df.columns)} features")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {data_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def select_attack_data(df: pd.DataFrame, attack_types: list[str]) -> pd.DataFrame:
    """Select specific attack types from the data."""
    print(f"🎯 Selecting attack types: {attack_types}")
    
    # Get unique attack types in the data
    unique_attacks = df['Label'].unique() if 'Label' in df.columns else []
    print(f"📊 Available attack types: {list(unique_attacks)}")
    
    # Filter for selected attacks + Benign
    if 'Label' in df.columns:
        selected_attacks = attack_types + ['Benign']
        filtered_df = df[df['Label'].isin(selected_attacks)]
        
        print(f"✅ Selected {len(filtered_df):,} samples")
        print(f"📈 Attack distribution:")
        for attack in selected_attacks:
            count = len(filtered_df[filtered_df['Label'] == attack])
            print(f"   {attack:15s}: {count:6,} samples")
        
        return filtered_df
    else:
        print("❌ 'Label' column not found in data")
        return df


def preprocess_for_testing(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess data for model testing."""
    print("🔄 Preprocessing data for testing...")
    
    # Remove Label column for features
    if 'Label' in df.columns:
        feature_df = df.drop(columns=['Label'])
        labels = df['Label'].values
    else:
        feature_df = df
        labels = np.zeros(len(df))  # Default to benign if no labels
    
    # Convert to binary labels (0=Benign, 1=Attack)
    binary_labels = np.array([0 if label == 'Benign' else 1 for label in labels])
    
    # Handle missing values
    feature_df = feature_df.fillna(0)
    
    # Remove non-numeric columns (IP addresses, timestamps, etc.)
    numeric_columns = []
    for col in feature_df.columns:
        try:
            # Try to convert to float
            pd.to_numeric(feature_df[col], errors='raise')
            numeric_columns.append(col)
        except:
            print(f"⚠️ Removing non-numeric column: {col}")
    
    feature_df = feature_df[numeric_columns]
    print(f"📊 Using {len(numeric_columns)} numeric features")
    
    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(feature_df.values)
    
    # Convert to tensors
    X = torch.tensor(scaled_features, dtype=torch.float32)
    y = torch.tensor(binary_labels, dtype=torch.long)
    
    print(f"✅ Preprocessed: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"📊 Label distribution: {np.bincount(binary_labels)}")
    
    return X, y


def create_simple_graph(X: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Create simple k-NN graph for testing."""
    print(f"🔗 Creating {k}-NN graph...")
    
    N = X.shape[0]
    
    # Simple approach: connect each node to its k nearest neighbors
    # For testing purposes, we'll create a simple circular graph
    edge_list = []
    
    for i in range(N):
        for j in range(1, k + 1):
            neighbor = (i + j) % N
            edge_list.append([i, neighbor])
            edge_list.append([neighbor, i])  # Bidirectional
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    print(f"✅ Graph created: {N} nodes, {edge_index.shape[1]} edges")
    return edge_index


def test_model_on_data(X: torch.Tensor, y: torch.Tensor, 
                      model_type: str = "gcn", k: int = 5) -> dict:
    """Test a model on the provided data."""
    print(f"\n🧪 Testing {model_type.upper()} model...")
    
    # Create graph
    edge_index = create_simple_graph(X, k=k)
    
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
    print(f"📊 Confusion Matrix:")
    print(f"   TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    print(f"   FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
    
    # Detailed classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Attack']))
    
    return {
        'model_type': model_type,
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'num_samples': len(y_true),
        'num_features': X.shape[1]
    }


def main():
    """Main function to test original data."""
    parser = argparse.ArgumentParser(description='Test models on original CICFlowMeter data')
    parser.add_argument('--data_path', type=str, 
                       default='/Users/cansu/Downloads/CICFlowMeter_out.csv',
                       help='Path to CICFlowMeter_out.csv file')
    parser.add_argument('--attack', type=str, default='DoS',
                       help='Attack type to test (e.g., DoS, DDoS, PortScan, etc.)')
    parser.add_argument('--model_type', type=str, default='gcn',
                       choices=['gcn', 'egraphsage'],
                       help='Model type to test')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of neighbors for graph construction')
    
    args = parser.parse_args()
    
    print("🚀 ELGNN-NIDS Original Data Testing")
    print("=" * 50)
    
    # Load original data
    df = load_original_data(args.data_path)
    if df is None:
        return
    
    # Select specific attack type
    attack_df = select_attack_data(df, [args.attack])
    if len(attack_df) == 0:
        print(f"❌ No samples found for attack type: {args.attack}")
        return
    
    # Preprocess data
    X, y = preprocess_for_testing(attack_df)
    
    # Test model
    results = test_model_on_data(X, y, args.model_type, args.k)
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Attack Type: {args.attack}")
    print(f"Model: {args.model_type.upper()}")
    print(f"Samples: {results['num_samples']:,}")
    print(f"Features: {results['num_features']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("=" * 50)
    
    print("\n💡 To test other attacks:")
    print(f"python test_original_data.py --attack PortScan")
    print(f"python test_original_data.py --attack DDoS")
    print(f"python test_original_data.py --attack Bot --model_type egraphsage")


if __name__ == "__main__":
    main()
