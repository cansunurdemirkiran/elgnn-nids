"""
evaluate.py

Comprehensive evaluation framework for ELGNN-NIDS models.
Supports model loading, performance evaluation, and result visualization.

Usage:
    python evaluate.py --model_path outputs/gcn_ewc_model.pt --dataset data2 --model_type gcn
    python evaluate.py --model_path outputs/egraphsage_ewc_model.pt --dataset data1 --model_type egraphsage
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

from graph_builder import load_graph
from src.config import features
from src.models import ELGNN, EGraphSAGE
from src.utils import banner, step


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate_dataset(self, dataset_name: str, split: str = 'test') -> Dict:
        """
        Evaluate model on a specific dataset.
        
        Args:
            dataset_name: 'data1' or 'data2'
            split: 'train', 'test', or 'val'
            
        Returns:
            Dictionary with comprehensive metrics
        """
        print(f"Evaluating on {dataset_name}/{split}...")
        
        # Load data and graph
        x, y, edge_index, edge_attr = load_graph(dataset_name, split)
        
        # Move to device
        x, y = x.to(self.device), y.to(self.device)
        edge_index = edge_index.to(self.device)
        
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            if edge_attr is not None:
                outputs = self.model(x, edge_index, edge_attr)
            else:
                outputs = self.model(x, edge_index)
            
            # Get probabilities and predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        # Convert to numpy for sklearn
        y_true = y.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        y_proba = probs.cpu().numpy()
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_proba)
        
        # Add dataset info
        metrics['dataset'] = dataset_name
        metrics['split'] = split
        metrics['num_samples'] = len(y_true)
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_proba: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        except ValueError:
            roc_auc = 0.0  # Handle case with only one class
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'specificity': specificity,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'confusion_matrix': cm.tolist()
        }
    
    def evaluate_all_tasks(self) -> Dict[str, Dict]:
        """Evaluate on both Task A and Task B."""
        results = {}
        
        # Evaluate on Task A
        results['task_a'] = self.evaluate_dataset('data1', 'test')
        
        # Evaluate on Task B
        results['task_b'] = self.evaluate_dataset('data2', 'test')
        
        return results


def load_model(model_path: str, model_type: str, input_dim: int) -> nn.Module:
    """Load trained model from checkpoint."""
    
    if model_type == 'gcn':
        model = ELGNN(input_dim=input_dim)
    elif model_type == 'egraphsage':
        model = EGraphSAGE(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         save_path: str, title: str = "Confusion Matrix"):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(results: Dict[str, Dict], save_path: str):
    """Plot metrics comparison between tasks."""
    
    # Extract metrics for plotting
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    task_a_values = [results['task_a'][m] for m in metrics]
    task_b_values = [results['task_b'][m] for m in metrics]
    
    # Create bar plot
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width/2, task_a_values, width, label='Task A', alpha=0.8)
    bars2 = ax.bar(x + width/2, task_b_values, width, label='Task B', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison: Task A vs Task B')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_evaluation_report(results: Dict[str, Dict], output_dir: Path):
    """Generate comprehensive evaluation report."""
    
    report = {
        'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'task_a_results': results['task_a'],
        'task_b_results': results['task_b'],
        'performance_comparison': {
            'accuracy_difference': results['task_b']['accuracy'] - results['task_a']['accuracy'],
            'f1_difference': results['task_b']['f1_score'] - results['task_a']['f1_score'],
            'backward_transfer': results['task_a']['accuracy']  # BWT = performance on Task A
        }
    }
    
    # Save detailed report
    report_file = output_dir / "evaluation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create summary text report
    summary_file = output_dir / "evaluation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("ELGNN-NIDS MODEL EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Evaluation Time: {report['evaluation_timestamp']}\n\n")
        
        f.write("TASK A RESULTS:\n")
        f.write("-" * 20 + "\n")
        for key, value in results['task_a'].items():
            if key != 'confusion_matrix':
                f.write(f"{key:25s}: {value:.4f}\n")
        f.write("\n")
        
        f.write("TASK B RESULTS:\n")
        f.write("-" * 20 + "\n")
        for key, value in results['task_b'].items():
            if key != 'confusion_matrix':
                f.write(f"{key:25s}: {value:.4f}\n")
        f.write("\n")
        
        f.write("PERFORMANCE COMPARISON:\n")
        f.write("-" * 25 + "\n")
        for key, value in report['performance_comparison'].items():
            f.write(f"{key:25s}: {value:.4f}\n")
    
    print(f"Evaluation report saved to {report_file}")
    print(f"Summary saved to {summary_file}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate ELGNN-NIDS models')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='both',
                       choices=['data1', 'data2', 'both'],
                       help='Dataset to evaluate')
    parser.add_argument('--model_type', type=str, default='gcn',
                       choices=['gcn', 'egraphsage'],
                       help='Model type')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    with banner(f"Model Evaluation - {args.model_type.upper()}"):
        
        # Load model
        with step("Loading model"):
            # Load a small sample to get input dimension
            try:
                x, _, _, _ = load_graph('data1', 'test')
                input_dim = x.shape[1]
            except FileNotFoundError:
                print("Warning: Graph files not found. Using default input_dim=44")
                input_dim = 44
            
            model = load_model(args.model_path, args.model_type, input_dim)
            print(f"Loaded {args.model_type} model from {args.model_path}")
        
        # Create evaluator
        evaluator = ModelEvaluator(model, device)
        
        # Evaluate based on dataset choice
        with step("Evaluating model"):
            if args.dataset == 'both':
                results = evaluator.evaluate_all_tasks()
            else:
                results = {
                    'single_task': evaluator.evaluate_dataset(args.dataset, 'test')
                }
        
        # Generate visualizations
        with step("Generating visualizations"):
            if args.dataset == 'both':
                # Confusion matrices
                plot_confusion_matrix(
                    results['task_a']['confusion_matrix'],
                    ['Benign', 'Attack'],
                    output_dir / "task_a_confusion_matrix.png",
                    "Task A Confusion Matrix"
                )
                
                plot_confusion_matrix(
                    results['task_b']['confusion_matrix'],
                    ['Benign', 'Attack'],
                    output_dir / "task_b_confusion_matrix.png",
                    "Task B Confusion Matrix"
                )
                
                # Metrics comparison
                plot_metrics_comparison(
                    results,
                    output_dir / "metrics_comparison.png"
                )
                
                # Generate report
                generate_evaluation_report(results, output_dir)
            else:
                # Single dataset evaluation
                single_results = results['single_task']
                plot_confusion_matrix(
                    single_results['confusion_matrix'],
                    ['Benign', 'Attack'],
                    output_dir / f"{args.dataset}_confusion_matrix.png",
                    f"{args.dataset.upper()} Confusion Matrix"
                )
                
                # Save single results
                single_file = output_dir / f"{args.dataset}_results.json"
                with open(single_file, 'w') as f:
                    json.dump(single_results, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        
        if args.dataset == 'both':
            print(f"Task A Accuracy: {results['task_a']['accuracy']:.4f}")
            print(f"Task B Accuracy: {results['task_b']['accuracy']:.4f}")
            print(f"Backward Transfer: {results['task_a']['accuracy']:.4f}")
        else:
            print(f"{args.dataset.upper()} Accuracy: {results['single_task']['accuracy']:.4f}")
            print(f"{args.dataset.upper()} F1-Score: {results['single_task']['f1_score']:.4f}")
            print(f"{args.dataset.upper()} ROC-AUC: {results['single_task']['roc_auc']:.4f}")
        
        print(f"Results saved to {output_dir}")
        print("=" * 50)


if __name__ == "__main__":
    main()
