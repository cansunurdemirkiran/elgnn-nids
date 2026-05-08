"""
train_models.py

Complete training script for ELGNN-NIDS models.
Supports all three training scenarios with proper data preparation and evaluation.

Usage:
    python train_models.py --scenario gcn_baseline --device cpu
    python train_models.py --scenario gcn_ewc --device cuda
    python train_models.py --scenario egraphsage_ewc --device cpu
"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.optim import Adam

from src.models import ELGNN, EGraphSAGE
from src.config import hyperparams as config
from graph_builder import load_graph
from ewc import EWC


def load_dataset_for_training(dataset_name: str, split: str):
    """Load dataset for training with proper error handling."""
    try:
        x, y, edge_index, edge_attr = load_graph(dataset_name, split)
        print(f"✅ Loaded {dataset_name}/{split}: {x.shape[0]} samples, {x.shape[1]} features")
        return x, y, edge_index, edge_attr
    except FileNotFoundError:
        print(f"❌ Graph files not found for {dataset_name}/{split}")
        print(f"💡 Run: python graph_builder.py")
        return None, None, None, None


def train_model(model, train_data, val_data, device, scenario: str, epochs: int = 50):
    """Train model with EWC if needed."""
    x_train, y_train, edge_index_train, edge_attr_train = train_data
    x_val, y_val, edge_index_val, edge_attr_val = val_data
    
    # Update model input dimension based on data
    input_dim = x_train.shape[1]
    if hasattr(model, 'conv1'):
        model.conv1.in_channels = input_dim
    if hasattr(model, 'sage1'):
        model.sage1.in_channels = input_dim
    
    model = model.to(device)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)
    edge_index_train = edge_index_train.to(device)
    edge_index_val = edge_index_val.to(device)
    
    if edge_attr_train is not None:
        edge_attr_train = edge_attr_train.to(device)
    if edge_attr_val is not None:
        edge_attr_val = edge_attr_val.to(device)
    
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    ewc = None
    if "ewc" in scenario:
        ewc = EWC(model, device)
    
    print(f"🏋️ Training {scenario} for {epochs} epochs...")
    print(f"📊 Train samples: {len(y_train)}, Val samples: {len(y_val)}")
    
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        
        if edge_attr_train is not None:
            outputs = model(x_train, edge_index_train, edge_attr_train)
        else:
            outputs = model(x_train, edge_index_train)
        
        loss = criterion(outputs, y_train)
        
        # Add EWC penalty if applicable
        if ewc is not None:
            ewc_loss = ewc.penalty(model)
            loss += config.EWC_LAMBDA * ewc_loss
        
        loss.backward()
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            if edge_attr_val is not None:
                val_outputs = model(x_val, edge_index_val, edge_attr_val)
            else:
                val_outputs = model(x_val, edge_index_val)
            
            val_loss = criterion(val_outputs, y_val)
            _, val_pred = torch.max(val_outputs, 1)
            val_acc = (val_pred == y_val).float().mean().item()
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}: Train Loss: {loss.item():.4f}, "
                  f"Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'outputs/{scenario}_best_model.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
    
    print(f"✅ Training completed. Best Val Acc: {best_val_acc:.4f}")
    return model, best_val_acc


def train_scenario(scenario: str, device: str = "cpu"):
    """Train specific scenario."""
    print(f"🚀 Starting {scenario} training on {device}")
    print("=" * 60)
    
    # Initialize model
    if "gcn" in scenario:
        model = ELGNN(
            input_dim=38,  # Will be updated dynamically based on loaded data
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            num_classes=config.NUM_CLASSES
        )
    elif "egraphsage" in scenario:
        model = EGraphSAGE(
            input_dim=38,  # Will be updated dynamically based on loaded data
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            num_classes=config.NUM_CLASSES
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    device_obj = torch.device(device)
    
    if scenario == "gcn_baseline":
        # Scenario A: Train only on Task B
        print("📚 Scenario A: GCN Baseline (Task B only)")
        
        train_data = load_dataset_for_training("data2", "train")
        val_data = load_dataset_for_training("data2", "test")
        
        if all(x is not None for x in train_data) and all(x is not None for x in val_data):
            model, acc = train_model(model, train_data, val_data, device_obj, scenario)
            
            # Save final model
            torch.save(model.state_dict(), f'outputs/{scenario}_model.pt')
            print(f"💾 Model saved: outputs/{scenario}_model.pt")
        
    elif scenario == "gcn_ewc":
        # Scenario B: GCN + EWC (Task A → Task B)
        print("📚 Scenario B: GCN + EWC (Task A → Task B)")
        
        # Phase 1: Train on Task A
        print("\n🔄 Phase 1: Training on Task A")
        train_data_a = load_dataset_for_training("data1", "train")
        val_data_a = load_dataset_for_training("data1", "test")
        
        if all(x is not None for x in train_data_a) and all(x is not None for x in val_data_a):
            model, acc_a = train_model(model, train_data_a, val_data_a, device_obj, scenario + "_phase1")
            
            # Store Task A weights for EWC
            ewc = EWC(model, device_obj)
            ewc.compute_fisher(model, train_data_a[0], train_data_a[1], train_data_a[2], train_data_a[3])
            print("✅ Task A weights stored for EWC")
        
        # Phase 2: Train on Task B with EWC
        print("\n🔄 Phase 2: Training on Task B with EWC")
        train_data_b = load_dataset_for_training("data2", "train")
        val_data_b = load_dataset_for_training("data2", "test")
        
        if all(x is not None for x in train_data_b) and all(x is not None for x in val_data_b):
            model, acc_b = train_model(model, train_data_b, val_data_b, device_obj, scenario + "_phase2", ewc)
            
            torch.save(model.state_dict(), f'outputs/{scenario}_model.pt')
            print(f"💾 Model saved: outputs/{scenario}_model.pt")
            print(f"📊 Task A Acc: {acc_a:.4f}, Task B Acc: {acc_b:.4f}")
    
    elif scenario == "egraphsage_ewc":
        # Scenario C: E-GraphSAGE + EWC (Task A → Task B)
        print("📚 Scenario C: E-GraphSAGE + EWC (Task A → Task B)")
        
        # Similar to Scenario B but with E-GraphSAGE
        # Phase 1: Train on Task A
        print("\n🔄 Phase 1: Training on Task A")
        train_data_a = load_dataset_for_training("data1", "train")
        val_data_a = load_dataset_for_training("data1", "test")
        
        if all(x is not None for x in train_data_a) and all(x is not None for x in val_data_a):
            model, acc_a = train_model(model, train_data_a, val_data_a, device_obj, scenario + "_phase1")
            
            ewc = EWC(model, device_obj)
            ewc.compute_fisher(model, train_data_a[0], train_data_a[1], train_data_a[2], train_data_a[3])
            print("✅ Task A weights stored for EWC")
        
        # Phase 2: Train on Task B with EWC
        print("\n🔄 Phase 2: Training on Task B with EWC")
        train_data_b = load_dataset_for_training("data2", "train")
        val_data_b = load_dataset_for_training("data2", "test")
        
        if all(x is not None for x in train_data_b) and all(x is not None for x in val_data_b):
            model, acc_b = train_model(model, train_data_b, val_data_b, device_obj, scenario + "_phase2", ewc)
            
            torch.save(model.state_dict(), f'outputs/{scenario}_model.pt')
            print(f"💾 Model saved: outputs/{scenario}_model.pt")
            print(f"📊 Task A Acc: {acc_a:.4f}, Task B Acc: {acc_b:.4f}")
    
    else:
        print(f"❌ Unknown scenario: {scenario}")
        return
    
    print("=" * 60)
    print(f"✅ {scenario} training completed!")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ELGNN-NIDS models')
    parser.add_argument('--scenario', type=str, required=True,
                       choices=['gcn_baseline', 'gcn_ewc', 'egraphsage_ewc'],
                       help='Training scenario')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Create outputs directory
    Path('outputs').mkdir(exist_ok=True)
    
    print("🚀 ELGNN-NIDS Model Training")
    print("=" * 60)
    print(f"Scenario: {args.scenario}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)
    
    # Train the specified scenario
    train_scenario(args.scenario, args.device)
    
    print("\n💡 Next steps:")
    print(f"1. Evaluate model: python evaluate.py --model_path outputs/{args.scenario}_model.pt")
    print(f"2. Test with original data: python tests/test_original_simple.py --model_path outputs/{args.scenario}_model.pt")
    print("3. Run SHAP analysis: python explain_shap.py")


if __name__ == "__main__":
    main()
