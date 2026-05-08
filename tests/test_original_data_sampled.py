"""
test_original_data_sampled.py

Efficient testing with original CICFlowMeter_out.csv data.
Samples from original data to match project's test dataset sizes.
Uses stratified sampling to maintain attack distribution.

Usage:
    python test_original_data_sampled.py --data_path /Users/cansu/Downloads/CICFlowMeter_out.csv --attack DoS --samples 5000
"""

from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from src.models import ELGNN, EGraphSAGE
from src.config import hyperparams as config


def load_and_sample_data(data_path: str, attack_type: str, 
                       total_samples: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load original data and sample efficiently."""
    print(f"📁 Loading original data from {data_path}")
    
    try:
        # Load data in chunks to avoid memory issues
        chunks = []
        chunk_size = 100000
        for chunk in pd.read_csv(data_path, chunksize=chunk_size):
            chunks.append(chunk)
            if len(chunks) * chunk_size > total_samples * 2:  # Stop early
                break
        
        df = pd.concat(chunks, ignore_index=True)
        print(f"✅ Loaded {len(df):,} samples from first chunks")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None
    
    # Get unique attack types
    if 'Label' not in df.columns:
        print("❌ 'Label' column not found")
        return None, None
    
    unique_attacks = df['Label'].unique()
    print(f"📊 Available attacks: {list(unique_attacks)}")
    
    # Filter for selected attack + Benign
    selected_attacks = [attack_type, 'Benign']
    filtered_df = df[df['Label'].isin(selected_attacks)]
    
    print(f"🎯 Filtered to {len(filtered_df):,} samples for {attack_type}")
    
    # Stratified sampling to maintain distribution
    attack_samples = filtered_df[filtered_df['Label'] == attack_type]
    benign_samples = filtered_df[filtered_df['Label'] == 'Benign']
    
    # Calculate sample sizes (70% attack, 30% benign like project)
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
    
    # Combine
    sampled_df = pd.concat([attack_sampled, benign_sampled], ignore_index=True)
    sampled_df = sampled_df.sample(frac=1.0, random_state=42)  # Shuffle
    
    print(f"📈 Sampled: {len(sampled_df)} total")
    print(f"   Attack samples: {len(attack_sampled)}")
    print(f"   Benign samples: {len(benign_sampled)}")
    
    return sampled_df, filtered_df


def preprocess_for_testing(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess data efficiently."""
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
    
    # Remove non-numeric columns efficiently
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
    
    return X, y


def create_efficient_graph(X: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Create k-NN graph efficiently."""
    print(f"🔗 Creating {k}-NN graph for {X.shape[0]} nodes...")
    
    N = X.shape[0]
    
    # Use sklearn for efficient k-NN
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(k+1, N), algorithm='auto')
    nbrs.fit(X.cpu().numpy())
    
    # Get k+1 nearest neighbors (including self)
    distances, indices = nbrs.kneighbors(X.cpu().numpy())
    
    # Remove self (first neighbor is always the point itself)
    indices = indices[:, 1:]  # Remove self
    distances = distances[:, 1:]
    
    # Create edge list
    edge_list = []
    for i in range(N):
        for j, neighbor_idx in enumerate(indices[i]):
            if neighbor_idx < N:  # Valid neighbor
                edge_list.append([i, neighbor_idx])
                edge_list.append([neighbor_idx, i])  # Bidirectional
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    print(f"✅ Graph: {N} nodes, {edge_index.shape[1]} edges")
    return edge_index


def test_model_efficient(X: torch.Tensor, y: torch.Tensor, 
                       model_type: str = "gcn") -> dict:
    """Test model efficiently."""
    print(f"\n🧪 Testing {model_type.upper()} model...")
    
    # Create simple graph for testing (connect each node to neighbors)
    edge_index = create_efficient_graph(X, k=5)  # Use smaller k for efficiency
    
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
    
    model.eval()
    
    # Test with smaller batches to avoid memory issues
    batch_size = 500
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            
            # Use only edges within the batch
            batch_mask = (edge_index[0] >= i) & (edge_index[0] < i+batch_size)
            if batch_mask.sum() == 0:  # No edges in this batch
                # Create simple sequential connections
                batch_size_actual = len(batch_X)
                edge_list = []
                for j in range(batch_size_actual - 1):
                    edge_list.append([j, j+1])
                    edge_list.append([j+1, j])
                batch_edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                batch_edge_index = edge_index[:, batch_mask]
                # Adjust indices to be relative to batch
                min_idx = batch_edge_index.min()
                batch_edge_index = batch_edge_index - min_idx
            
            # Forward pass
            outputs = model(batch_X, batch_edge_index)
            _, predicted = torch.max(outputs, 1)
            all_predictions.append(predicted)
    
    # Combine predictions
    y_pred = torch.cat(all_predictions).numpy()
    y_true = y.numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"✅ {model_type.upper()} Accuracy: {accuracy:.4f}")
    
    return {
        'model_type': model_type,
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'num_samples': len(y_true)
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Efficient testing with original data')
    parser.add_argument('--data_path', type=str, 
                       default='/Users/cansu/Downloads/CICFlowMeter_out.csv',
                       help='Path to CICFlowMeter_out.csv')
    parser.add_argument('--attack', type=str, default='DoS',
                       help='Attack type to test')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Total samples to test')
    parser.add_argument('--model_type', type=str, default='gcn',
                       choices=['gcn', 'egraphsage'],
                       help='Model type')
    
    args = parser.parse_args()
    
    print("🚀 ELGNN-NIDS Efficient Original Data Testing")
    print("=" * 60)
    
    # Load and sample data
    sampled_df, full_df = load_and_sample_data(args.data_path, args.attack, args.samples)
    if sampled_df is None:
        return
    
    # Preprocess
    X, y = preprocess_for_testing(sampled_df)
    
    # Test model
    results = test_model_efficient(X, y, args.model_type)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Attack Type: {args.attack}")
    print(f"Model: {args.model_type.upper()}")
    print(f"Total Samples: {results['num_samples']:,}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    
    # Confusion matrix details
    cm = results['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()
    print(f"True Positives: {tp:,}")
    print(f"True Negatives: {tn:,}")
    print(f"False Positives: {fp:,}")
    print(f"False Negatives: {fn:,}")
    
    print("=" * 60)
    print("✅ Testing completed efficiently!")


if __name__ == "__main__":
    main()
