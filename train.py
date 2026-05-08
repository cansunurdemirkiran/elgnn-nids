"""
train.py

Training script for ELGNN-NIDS project supporting three scenarios:
1. GCN baseline (Task B only)
2. GCN + EWC (Task A → Task B with continual learning)
3. E-GraphSAGE + EWC (Task A → Task B with continual learning)

Usage:
    python train.py --scenario gcn_baseline
    python train.py --scenario gcn_ewc
    python train.py --scenario e_graphsage_ewc
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from graph_builder import load_graph
from src.config import hyperparams as config
from src.data import load_dataset
from src.models import ELGNN, EGraphSAGE
from src.utils import banner, step
from ewc import EWC


class Trainer:
    """Training class for GNN models."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, x: torch.Tensor, y: torch.Tensor, 
                   edge_index: torch.Tensor, edge_attr: torch.Tensor = None,
                   ewc: EWC = None) -> float:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        if edge_attr is not None:
            outputs = self.model(x, edge_index, edge_attr)
        else:
            outputs = self.model(x, edge_index)
        
        # Compute loss
        if ewc is not None:
            loss = ewc.total_loss(self.model, outputs, y)
        else:
            loss = self.criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, x: torch.Tensor, y: torch.Tensor, 
                edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> Dict[str, float]:
        """Evaluate model on test data."""
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            if edge_attr is not None:
                outputs = self.model(x, edge_index, edge_attr)
            else:
                outputs = self.model(x, edge_index)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Move to CPU for sklearn metrics
            y_true = y.cpu().numpy()
            y_pred = predicted.cpu().numpy()
            
            # Compute metrics
            accuracy = accuracy_score(y_true, y_pred)
            
            return {
                'accuracy': accuracy,
                'loss': self.criterion(outputs, y).item(),
                'predictions': y_pred,
                'true_labels': y_true
            }


def train_gcn_baseline(device: torch.device) -> Dict[str, float]:
    """Scenario A: GCN baseline trained only on Task B."""
    
    with banner("Scenario A: GCN Baseline (Task B Only)"):
        # Load Task B data
        with step("Loading Task B data"):
            x_train, y_train, edge_index_train, edge_attr_train = load_graph('data2', 'train')
            x_test, y_test, edge_index_test, edge_attr_test = load_graph('data2', 'test')
            
            # Move to device
            x_train, y_train = x_train.to(device), y_train.to(device)
            x_test, y_test = x_test.to(device), y_test.to(device)
            edge_index_train = edge_index_train.to(device)
            edge_index_test = edge_index_test.to(device)
            
            print(f"Task B train: {x_train.shape[0]} samples")
            print(f"Task B test: {x_test.shape[0]} samples")
        
        # Initialize model
        with step("Initializing GCN model"):
            model = ELGNN(
                input_dim=x_train.shape[1],
                hidden_dim=config.HIDDEN_DIM,
                output_dim=config.OUTPUT_DIM,
                num_classes=config.NUM_CLASSES
            )
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        with step("Training GCN model"):
            trainer = Trainer(model, device)
            
            for epoch in tqdm(range(config.NUM_EPOCHS), desc="Training"):
                loss = trainer.train_epoch(x_train, y_train, edge_index_train)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:3d}, Loss: {loss:.4f}")
        
        # Evaluate on both datasets
        with step("Evaluating model"):
            # Task B evaluation
            results_b = trainer.evaluate(x_test, y_test, edge_index_test)
            
            # Task A evaluation (to measure backward transfer)
            x_test_a, y_test_a, edge_index_a, edge_attr_a = load_graph('data1', 'test')
            x_test_a, y_test_a = x_test_a.to(device), y_test_a.to(device)
            edge_index_a = edge_index_a.to(device)
            results_a = trainer.evaluate(x_test_a, y_test_a, edge_index_a)
            
            print(f"Task B Accuracy: {results_b['accuracy']:.4f}")
            print(f"Task A Accuracy: {results_a['accuracy']:.4f}")
        
        return {
            'task_b_accuracy': results_b['accuracy'],
            'task_a_accuracy': results_a['accuracy'],
            'backward_transfer': results_a['accuracy']  # BWT = performance on Task A after Task B
        }


def train_gcn_ewc(device: torch.device) -> Dict[str, float]:
    """Scenario B: GCN + EWC continual learning (Task A → Task B)."""
    
    with banner("Scenario B: GCN + EWC Continual Learning"):
        # Load Task A data
        with step("Loading Task A data"):
            x_train_a, y_train_a, edge_index_a, edge_attr_a = load_graph('data1', 'train')
            x_test_a, y_test_a, edge_index_test_a, edge_attr_test_a = load_graph('data1', 'test')
            
            # Move to device
            x_train_a, y_train_a = x_train_a.to(device), y_train_a.to(device)
            x_test_a, y_test_a = x_test_a.to(device), y_test_a.to(device)
            edge_index_a = edge_index_a.to(device)
            edge_index_test_a = edge_index_test_a.to(device)
            
            print(f"Task A train: {x_train_a.shape[0]} samples")
            print(f"Task A test: {x_test_a.shape[0]} samples")
        
        # Initialize and train on Task A
        with step("Training on Task A"):
            model = ELGNN(
                input_dim=x_train_a.shape[1],
                hidden_dim=config.HIDDEN_DIM,
                output_dim=config.OUTPUT_DIM,
                num_classes=config.NUM_CLASSES
            )
            trainer = Trainer(model, device)
            
            for epoch in tqdm(range(config.NUM_EPOCHS), desc="Task A Training"):
                loss = trainer.train_epoch(x_train_a, y_train_a, edge_index_a)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:3d}, Loss: {loss:.4f}")
        
        # Evaluate Task A performance
        with step("Evaluating Task A performance"):
            results_a_before = trainer.evaluate(x_test_a, y_test_a, edge_index_test_a)
            task_a_performance_before = results_a_before['accuracy']
            print(f"Task A Accuracy (before Task B): {task_a_performance_before:.4f}")
        
        # Create EWC instance
        with step("Computing Fisher information"):
            dataset_a = TensorDataset(x_train_a, y_train_a)
            dataloader_a = DataLoader(dataset_a, batch_size=32, shuffle=False)
            ewc = EWC(model, dataloader_a, lambda_ewc=config.EWC_LAMBDA)
        
        # Load Task B data
        with step("Loading Task B data"):
            x_train_b, y_train_b, edge_index_b, edge_attr_b = load_graph('data2', 'train')
            x_test_b, y_test_b, edge_index_test_b, edge_attr_test_b = load_graph('data2', 'test')
            
            # Move to device
            x_train_b, y_train_b = x_train_b.to(device), y_train_b.to(device)
            x_test_b, y_test_b = x_test_b.to(device), y_test_b.to(device)
            edge_index_b = edge_index_b.to(device)
            edge_index_test_b = edge_index_test_b.to(device)
            
            print(f"Task B train: {x_train_b.shape[0]} samples")
            print(f"Task B test: {x_test_b.shape[0]} samples")
        
        # Train on Task B with EWC
        with step("Training on Task B with EWC"):
            for epoch in tqdm(range(config.NUM_EPOCHS), desc="Task B Training"):
                loss = trainer.train_epoch(x_train_b, y_train_b, edge_index_b, ewc=ewc)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:3d}, Loss: {loss:.4f}")
        
        # Final evaluation
        with step("Final evaluation"):
            # Task A evaluation (after Task B)
            results_a_after = trainer.evaluate(x_test_a, y_test_a, edge_index_test_a)
            task_a_performance_after = results_a_after['accuracy']
            
            # Task B evaluation
            results_b = trainer.evaluate(x_test_b, y_test_b, edge_index_test_b)
            
            # Calculate metrics
            backward_transfer = task_a_performance_after - task_a_performance_before
            
            print(f"Task A Accuracy (before): {task_a_performance_before:.4f}")
            print(f"Task A Accuracy (after):  {task_a_performance_after:.4f}")
            print(f"Task B Accuracy: {results_b['accuracy']:.4f}")
            print(f"Backward Transfer (BWT): {backward_transfer:.4f}")
        
        return {
            'task_a_accuracy_before': task_a_performance_before,
            'task_a_accuracy_after': task_a_performance_after,
            'task_b_accuracy': results_b['accuracy'],
            'backward_transfer': backward_transfer
        }


def train_egraphsage_ewc(device: torch.device) -> Dict[str, float]:
    """Scenario C: E-GraphSAGE + EWC continual learning (Task A → Task B)."""
    
    with banner("Scenario C: E-GraphSAGE + EWC Continual Learning"):
        # Load Task A data
        with step("Loading Task A data"):
            x_train_a, y_train_a, edge_index_a, edge_attr_a = load_graph('data1', 'train')
            x_test_a, y_test_a, edge_index_test_a, edge_attr_test_a = load_graph('data1', 'test')
            
            # Move to device
            x_train_a, y_train_a = x_train_a.to(device), y_train_a.to(device)
            x_test_a, y_test_a = x_test_a.to(device), y_test_a.to(device)
            edge_index_a = edge_index_a.to(device)
            edge_attr_a = edge_attr_a.to(device)
            edge_index_test_a = edge_index_test_a.to(device)
            edge_attr_test_a = edge_attr_test_a.to(device)
            
            print(f"Task A train: {x_train_a.shape[0]} samples")
            print(f"Task A test: {x_test_a.shape[0]} samples")
        
        # Initialize and train on Task A
        with step("Training E-GraphSAGE on Task A"):
            model = EGraphSAGE(
                input_dim=x_train_a.shape[1],
                hidden_dim=config.HIDDEN_DIM,
                output_dim=config.OUTPUT_DIM,
                num_classes=config.NUM_CLASSES
            )
            trainer = Trainer(model, device)
            
            for epoch in tqdm(range(config.NUM_EPOCHS), desc="Task A Training"):
                loss = trainer.train_epoch(x_train_a, y_train_a, edge_index_a, edge_attr_a)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:3d}, Loss: {loss:.4f}")
        
        # Evaluate Task A performance
        with step("Evaluating Task A performance"):
            results_a_before = trainer.evaluate(x_test_a, y_test_a, edge_index_test_a, edge_attr_test_a)
            task_a_performance_before = results_a_before['accuracy']
            print(f"Task A Accuracy (before Task B): {task_a_performance_before:.4f}")
        
        # Create EWC instance
        with step("Computing Fisher information"):
            dataset_a = TensorDataset(x_train_a, y_train_a)
            dataloader_a = DataLoader(dataset_a, batch_size=32, shuffle=False)
            ewc = EWC(model, dataloader_a, lambda_ewc=config.EWC_LAMBDA)
        
        # Load Task B data
        with step("Loading Task B data"):
            x_train_b, y_train_b, edge_index_b, edge_attr_b = load_graph('data2', 'train')
            x_test_b, y_test_b, edge_index_test_b, edge_attr_test_b = load_graph('data2', 'test')
            
            # Move to device
            x_train_b, y_train_b = x_train_b.to(device), y_train_b.to(device)
            x_test_b, y_test_b = x_test_b.to(device), y_test_b.to(device)
            edge_index_b = edge_index_b.to(device)
            edge_attr_b = edge_attr_b.to(device)
            edge_index_test_b = edge_index_test_b.to(device)
            edge_attr_test_b = edge_attr_test_b.to(device)
            
            print(f"Task B train: {x_train_b.shape[0]} samples")
            print(f"Task B test: {x_test_b.shape[0]} samples")
        
        # Train on Task B with EWC
        with step("Training on Task B with EWC"):
            for epoch in tqdm(range(config.NUM_EPOCHS), desc="Task B Training"):
                loss = trainer.train_epoch(x_train_b, y_train_b, edge_index_b, edge_attr_b, ewc=ewc)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:3d}, Loss: {loss:.4f}")
        
        # Final evaluation
        with step("Final evaluation"):
            # Task A evaluation (after Task B)
            results_a_after = trainer.evaluate(x_test_a, y_test_a, edge_index_test_a, edge_attr_test_a)
            task_a_performance_after = results_a_after['accuracy']
            
            # Task B evaluation
            results_b = trainer.evaluate(x_test_b, y_test_b, edge_index_test_b, edge_attr_test_b)
            
            # Calculate metrics
            backward_transfer = task_a_performance_after - task_a_performance_before
            
            print(f"Task A Accuracy (before): {task_a_performance_before:.4f}")
            print(f"Task A Accuracy (after):  {task_a_performance_after:.4f}")
            print(f"Task B Accuracy: {results_b['accuracy']:.4f}")
            print(f"Backward Transfer (BWT): {backward_transfer:.4f}")
        
        return {
            'task_a_accuracy_before': task_a_performance_before,
            'task_a_accuracy_after': task_a_performance_after,
            'task_b_accuracy': results_b['accuracy'],
            'backward_transfer': backward_transfer
        }


def save_results(results: Dict[str, float], scenario: str):
    """Save training results to file."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / f"{scenario}_results.txt"
    
    with open(results_file, 'w') as f:
        f.write(f"Results for {scenario}\n")
        f.write("=" * 50 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")
    
    print(f"Results saved to {results_file}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ELGNN-NIDS models')
    parser.add_argument('--scenario', type=str, required=True,
                       choices=['gcn_baseline', 'gcn_ewc', 'egraphsage_ewc'],
                       help='Training scenario')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Train based on scenario
    start_time = time.time()
    
    if args.scenario == 'gcn_baseline':
        results = train_gcn_baseline(device)
    elif args.scenario == 'gcn_ewc':
        results = train_gcn_ewc(device)
    elif args.scenario == 'egraphsage_ewc':
        results = train_egraphsage_ewc(device)
    
    # Save results
    save_results(results, args.scenario)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print("=" * 50)


if __name__ == "__main__":
    main()
