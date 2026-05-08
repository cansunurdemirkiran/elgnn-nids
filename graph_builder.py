"""
graph_builder.py

k-NN graph construction for network intrusion detection.
Each node represents a network flow, edges connect k nearest neighbors based on Euclidean distance.

Public API:
    build_graph(X, k=10) -> edge_index, edge_attr
    build_all_graphs() -> dict with graphs for all datasets
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from src.config import paths
from src.data import load_dataset
from src.utils import banner, step


def build_graph(X: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build k-NN graph from feature matrix.
    
    Args:
        X: (N, F) feature tensor
        k: Number of nearest neighbors
        
    Returns:
        edge_index: (2, E) edge list in PyG COO format
        edge_attr: (E, F) edge features (absolute difference between node features)
    """
    N, F = X.shape
    
    # Convert to numpy for sklearn
    X_np = X.cpu().numpy()
    
    # Build k-NN graph
    print(f"Building {k}-NN graph for {N} nodes...")
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric='euclidean')
    nbrs.fit(X_np)
    
    # Get k+1 nearest neighbors (including self)
    distances, indices = nbrs.kneighbors(X_np)
    
    # Remove self-loops (first neighbor is always the node itself)
    indices = indices[:, 1:]  # (N, k)
    distances = distances[:, 1:]  # (N, k)
    
    # Create edge list
    edge_list = []
    edge_features = []
    
    for i in range(N):
        for j, neighbor_idx in enumerate(indices[i]):
            # Add edge in both directions for undirected graph
            edge_list.append([i, neighbor_idx])
            edge_list.append([neighbor_idx, i])
            
            # Edge feature: absolute difference between node features
            edge_feat = np.abs(X_np[i] - X_np[neighbor_idx])
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)  # Same feature for reverse edge
    
    # Convert to tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    
    print(f"Graph built: {N} nodes, {edge_index.shape[1]} edges")
    return edge_index, edge_attr


def build_all_graphs(k: int = 10, save_graphs: bool = True) -> dict:
    """
    Build graphs for all datasets (data1 and data2, train and test splits).
    
    Args:
        k: Number of nearest neighbors
        save_graphs: Whether to save graphs to disk
        
    Returns:
        Dictionary containing graphs for all dataset splits
    """
    graphs = {}
    output_dir = Path("graphs")
    output_dir.mkdir(exist_ok=True)
    
    print("="*50)
    print("Building k-NN Graphs")
    print("="*50)
    
    for dataset_name in paths.DATASETS:
        for split in ["train", "test"]:
            print(f"Processing {dataset_name}/{split}...")
            
            # Load dataset
            dataset = load_dataset(dataset_name, split, verbose=False)
            
            # Build graph
            edge_index, edge_attr = build_graph(dataset.X, k=k)
            
            # Store graph information
            graph_key = f"{dataset_name}_{split}"
            graphs[graph_key] = {
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "num_nodes": dataset.X.shape[0],
                "num_features": dataset.X.shape[1],
                "num_edges": edge_index.shape[1],
            }
            
            # Save graphs if requested
            if save_graphs:
                graph_file = output_dir / f"{graph_key}_graph.pt"
                # Handle both TrainSplit and EvalSplit
                y_data = dataset.y if hasattr(dataset, 'y') else dataset.y_binary
                torch.save({
                    "edge_index": edge_index,
                    "edge_attr": edge_attr,
                    "x": dataset.X,
                    "y": y_data,
                }, graph_file)
                print(f"Saved graph to {graph_file}")
    
    return graphs


def load_graph(dataset_name: str, split: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load pre-computed graph from disk.
    
    Args:
        dataset_name: 'data1' or 'data2'
        split: 'train' or 'test'
        
    Returns:
        x: (N, F) node features
        y: (N,) labels
        edge_index: (2, E) edge list
        edge_attr: (E, F) edge features
    """
    graph_file = Path("graphs") / f"{dataset_name}_{split}_graph.pt"
    
    if not graph_file.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_file}. Run build_all_graphs() first.")
    
    data = torch.load(graph_file)
    return data["x"], data["y"], data["edge_index"], data["edge_attr"]


def main():
    """Main function to build all graphs."""
    start_time = time.time()
    
    # Build graphs for all datasets
    graphs = build_all_graphs(k=10, save_graphs=True)
    
    # Print summary
    print("\n" + "="*50)
    print("GRAPH BUILDING SUMMARY")
    print("="*50)
    
    total_edges = 0
    for graph_key, graph_info in graphs.items():
        print(f"{graph_key:15s}: {graph_info['num_nodes']:6d} nodes, {graph_info['num_edges']:8d} edges")
        total_edges += graph_info['num_edges']
    
    print(f"{'Total':15s}: {'-':>6s} nodes, {total_edges:8d} edges")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print("="*50)


if __name__ == "__main__":
    main()
