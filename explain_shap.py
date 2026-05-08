"""
explain_shap.py

SHAP (SHapley Additive exPlanations) analysis for model interpretability.
This script computes feature importance for trained models and generates visualizations.

Usage:
    python explain_shap.py --model_path models/gcn_taskB_alone.pt --dataset data2
    python explain_shap.py --model_path models/egraphsage_ewc.pt --dataset data1
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
import shap
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from graph_builder import load_graph
from src.config import features
from src.data import load_dataset
from src.models import ELGNN, EGraphSAGE
from src.utils import banner, step


class SHAPExplainer:
    """SHAP explainer for GNN models."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor, 
                      edge_attr: torch.Tensor = None) -> np.ndarray:
        """
        Predict probabilities for SHAP explainer.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features (optional)
            
        Returns:
            Probability predictions
        """
        with torch.no_grad():
            if edge_attr is not None:
                logits = self.model(x, edge_index, edge_attr)
            else:
                logits = self.model(x, edge_index)
            
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()
    
    def explain_samples(self, x: torch.Tensor, edge_index: torch.Tensor,
                       edge_attr: torch.Tensor = None, sample_indices: List[int] = None,
                       max_samples: int = 100) -> Dict:
        """
        Generate SHAP explanations for specified samples.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features (optional)
            sample_indices: Specific sample indices to explain
            max_samples: Maximum number of samples to explain
            
        Returns:
            Dictionary containing SHAP values and metadata
        """
        if sample_indices is None:
            # Randomly sample indices
            n_samples = min(max_samples, x.shape[0])
            sample_indices = np.random.choice(x.shape[0], n_samples, replace=False)
        
        print(f"Explaining {len(sample_indices)} samples...")
        
        # Prepare data for SHAP
        x_np = x.cpu().numpy()
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(
            lambda X: self.predict_proba(
                torch.tensor(X, dtype=torch.float32).to(self.device),
                edge_index,
                edge_attr
            ),
            x_np[:100]  # Use first 100 samples as background
        )
        
        # Explain selected samples
        shap_values = []
        explanations = []
        
        for i, idx in enumerate(tqdm(sample_indices, desc="Computing SHAP values")):
            # Get SHAP values for this sample
            sample = x_np[idx:idx+1]
            sv = explainer.shap_values(sample, nsamples=100)
            
            # Store values (focus on class 1 - attack)
            if isinstance(sv, list):
                shap_values.append(sv[1])  # Class 1 (attack)
            else:
                shap_values.append(sv)
            
            explanations.append({
                'sample_index': int(idx),
                'true_label': int(edge_index[0][idx].item()) if edge_index is not None else 0,
                'predicted_prob': float(self.predict_proba(
                    torch.tensor(sample, dtype=torch.float32).to(self.device),
                    edge_index,
                    edge_attr
                )[0][1])
            })
        
        return {
            'shap_values': np.array(shap_values),
            'feature_names': self._get_feature_names(),
            'explanations': explanations,
            'sample_indices': sample_indices.tolist()
        }
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for explanations."""
        # This would need to be implemented based on your feature preprocessing
        # For now, return generic names
        return [f"feature_{i}" for i in range(44)]  # Assuming 44 features


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


def load_selected_samples(dataset_name: str) -> Dict:
    """Load pre-selected SHAP samples."""
    samples_file = Path("xai_outputs") / dataset_name / "selected_samples.json"
    
    if not samples_file.exists():
        raise FileNotFoundError(f"Selected samples not found: {samples_file}. Run sample selection first.")
    
    with open(samples_file, 'r') as f:
        return json.load(f)


def plot_feature_importance(shap_values: np.ndarray, feature_names: List[str], 
                          save_path: str, title: str = "Feature Importance"):
    """Plot feature importance using SHAP values."""
    
    # Calculate mean absolute SHAP values for each feature
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df.head(20), x='importance', y='feature')
    plt.title(f"{title} - Top 20 Features")
    plt.xlabel("Mean |SHAP Value|")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_df


def plot_shap_summary(shap_values: np.ndarray, feature_names: List[str], 
                     save_path: str, max_display: int = 20):
    """Create SHAP summary plot."""
    
    # Limit to top features
    if len(feature_names) > max_display:
        importance = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(importance)[-max_display:]
        shap_values = shap_values[:, top_indices]
        feature_names = [feature_names[i] for i in top_indices]
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, feature_names=feature_names, 
                      plot_type="bar", show=False, max_display=max_display)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def compare_models_across_tasks(model_results: Dict[str, Dict], output_dir: Path):
    """Compare feature importance across different models and tasks."""
    
    # Collect all feature importances
    all_importances = {}
    
    for model_name, results in model_results.items():
        shap_values = results['shap_values']
        feature_names = results['feature_names']
        
        # Calculate mean importance
        mean_importance = np.mean(np.abs(shap_values), axis=0)
        
        all_importances[model_name] = dict(zip(feature_names, mean_importance))
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_importances).T
    
    # Plot comparison heatmap
    plt.figure(figsize=(16, 10))
    sns.heatmap(comparison_df, cmap='viridis', annot=False)
    plt.title("Feature Importance Comparison Across Models")
    plt.xlabel("Features")
    plt.ylabel("Models")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comparison table
    comparison_df.to_csv(output_dir / "feature_importance_comparison.csv")
    
    return comparison_df


def main():
    """Main SHAP analysis function."""
    parser = argparse.ArgumentParser(description='SHAP analysis for ELGNN-NIDS models')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, required=True, choices=['data1', 'data2'],
                       help='Dataset to analyze')
    parser.add_argument('--model_type', type=str, default='gcn', choices=['gcn', 'egraphsage'],
                       help='Model type')
    parser.add_argument('--output_dir', type=str, default='xai_outputs',
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum number of samples to explain')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with banner(f"SHAP Analysis - {args.model_type.upper()} on {args.dataset.upper()}"):
        
        # Load data
        with step("Loading data and graph"):
            x, y, edge_index, edge_attr = load_graph(args.dataset, 'test')
            print(f"Loaded {x.shape[0]} samples with {x.shape[1]} features")
        
        # Load model
        with step("Loading model"):
            model = load_model(args.model_path, args.model_type, x.shape[1])
            print(f"Loaded {args.model_type} model from {args.model_path}")
        
        # Load selected samples
        with step("Loading selected samples"):
            try:
                selected_samples = load_selected_samples(args.dataset)
                sample_indices = [s['sample_index'] for s in selected_samples['explained_samples']]
                print(f"Using {len(sample_indices)} pre-selected samples")
            except FileNotFoundError:
                print("No pre-selected samples found, using random selection")
                sample_indices = None
        
        # Create SHAP explainer
        with step("Computing SHAP explanations"):
            explainer = SHAPExplainer(model, device)
            
            # Determine if edge_attr is needed
            use_edge_attr = edge_attr is not None and args.model_type == 'egraphsage'
            
            results = explainer.explain_samples(
                x, edge_index, 
                edge_attr if use_edge_attr else None,
                sample_indices=sample_indices,
                max_samples=args.max_samples
            )
            
            print(f"Computed SHAP values for {len(results['explanations'])} samples")
        
        # Generate visualizations
        with step("Generating visualizations"):
            # Feature importance plot
            importance_df = plot_feature_importance(
                results['shap_values'], 
                results['feature_names'],
                output_dir / f"{args.model_type}_feature_importance.png",
                f"{args.model_type.upper()} - {args.dataset.upper()} Feature Importance"
            )
            
            # SHAP summary plot
            plot_shap_summary(
                results['shap_values'],
                results['feature_names'],
                output_dir / f"{args.model_type}_shap_summary.png"
            )
            
            print(f"Saved visualizations to {output_dir}")
        
        # Save results
        with step("Saving results"):
            results_file = output_dir / f"{args.model_type}_shap_results.json"
            
            # Convert numpy arrays to lists for JSON serialization
            save_results = {
                'model_type': args.model_type,
                'dataset': args.dataset,
                'sample_count': len(results['explanations']),
                'feature_count': len(results['feature_names']),
                'feature_names': results['feature_names'],
                'explanations': results['explanations'],
                'mean_shap_values': np.mean(np.abs(results['shap_values']), axis=0).tolist(),
                'std_shap_values': np.std(results['shap_values'], axis=0).tolist()
            }
            
            with open(results_file, 'w') as f:
                json.dump(save_results, f, indent=2)
            
            print(f"Saved results to {results_file}")
    
    print(f"\nSHAP analysis completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
