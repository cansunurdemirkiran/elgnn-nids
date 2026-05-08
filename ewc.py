"""
ewc.py

Elastic Weight Consolidation (EWC) implementation for continual learning.
EWC prevents catastrophic forgetting by adding a penalty term to the loss function
that preserves important weights from previous tasks.

Reference: Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks"

Public API:
    EWC(model, dataset, lambda_ewc=100) -> EWC instance
    compute_fisher_matrix(model, dataset) -> dict of Fisher information
    ewc_loss(model, fisher_info, optimal_params, lambda_ewc) -> penalty term
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.config import hyperparams as config


class EWC:
    """
    Elastic Weight Consolidation for continual learning.
    
    Attributes:
        model: The neural network model
        fisher_info: Dictionary containing Fisher information for each parameter
        optimal_params: Dictionary containing optimal parameters from previous task
        lambda_ewc: EWC regularization strength
    """
    
    def __init__(self, model: nn.Module, dataset: DataLoader, lambda_ewc: float = config.EWC_LAMBDA):
        """
        Initialize EWC with a trained model and dataset.
        
        Args:
            model: Trained neural network model
            dataset: DataLoader containing the previous task data
            lambda_ewc: EWC regularization strength (default: 100)
        """
        self.model = model
        self.lambda_ewc = lambda_ewc
        
        # Store optimal parameters
        self.optimal_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone().detach()
        
        # Compute Fisher information matrix
        print("Computing Fisher information matrix...")
        self.fisher_info = self.compute_fisher_matrix(model, dataset)
        print("Fisher information computation completed.")
    
    def compute_fisher_matrix(self, model: nn.Module, dataset: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher information matrix for each parameter.
        
        Args:
            model: Neural network model
            dataset: DataLoader containing the data
            
        Returns:
            Dictionary containing Fisher information for each parameter
        """
        model.eval()
        fisher_info = {}
        
        # Initialize Fisher information
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param.data)
        
        # Compute Fisher information using the dataset
        num_samples = 0
        for batch_x, batch_y in tqdm(dataset, desc="Computing Fisher"):
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            
            # Forward pass
            model.zero_grad()
            outputs = model(batch_x)
            loss = F.cross_entropy(outputs, batch_y)
            
            # Backward pass to get gradients
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
            
            num_samples += batch_x.size(0)
        
        # Normalize by number of samples
        for name in fisher_info:
            fisher_info[name] /= num_samples
        
        return fisher_info
    
    def ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty term.
        
        Args:
            model: Current model parameters
            
        Returns:
            EWC penalty term
        """
        penalty = 0.0
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.fisher_info:
                # EWC penalty: λ * F * (θ - θ*)^2
                penalty += (self.fisher_info[name] * (param - self.optimal_params[name]) ** 2).sum()
        
        return self.lambda_ewc * penalty
    
    def total_loss(self, model: nn.Module, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute total loss including EWC penalty.
        
        Args:
            model: Neural network model
            outputs: Model predictions
            targets: True labels
            
        Returns:
            Total loss (cross entropy + EWC penalty)
        """
        ce_loss = F.cross_entropy(outputs, targets)
        ewc_penalty = self.ewc_loss(model)
        
        return ce_loss + ewc_penalty


def compute_fisher_matrix(model: nn.Module, dataset: DataLoader) -> Dict[str, torch.Tensor]:
    """
    Standalone function to compute Fisher information matrix.
    
    Args:
        model: Neural network model
        dataset: DataLoader containing the data
        
    Returns:
        Dictionary containing Fisher information for each parameter
    """
    ewc = EWC(model, dataset, lambda_ewc=0.0)  # Initialize with lambda=0 to avoid penalty
    return ewc.fisher_info


def ewc_loss(model: nn.Module, fisher_info: Dict[str, torch.Tensor], 
             optimal_params: Dict[str, torch.Tensor], lambda_ewc: float = config.EWC_LAMBDA) -> torch.Tensor:
    """
    Standalone function to compute EWC penalty.
    
    Args:
        model: Current model parameters
        fisher_info: Fisher information matrix
        optimal_params: Optimal parameters from previous task
        lambda_ewc: EWC regularization strength
        
    Returns:
        EWC penalty term
    """
    penalty = 0.0
    
    for name, param in model.named_parameters():
        if param.requires_grad and name in fisher_info:
            penalty += (fisher_info[name] * (param - optimal_params[name]) ** 2).sum()
    
    return lambda_ewc * penalty


def create_dataloader(x: torch.Tensor, y: torch.Tensor, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader from tensors.
    
    Args:
        x: Feature tensor
        y: Label tensor
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader instance
    """
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def main():
    """Example usage of EWC."""
    # This is just an example - actual usage will be in train.py
    print("EWC module loaded successfully!")
    print(f"Default EWC lambda: {config.EWC_LAMBDA}")
    print("Use EWC in train.py for continual learning scenarios.")


if __name__ == "__main__":
    main()
