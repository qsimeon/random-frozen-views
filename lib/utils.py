"""Utility functions for frozen view contrastive learning.

This module provides helper functions for data processing, visualization,
and evaluation of contrastive learning models using frozen random projections.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_similarity_matrix(
    features1: torch.Tensor,
    features2: Optional[torch.Tensor] = None,
    normalize: bool = True,
    metric: str = 'cosine'
) -> torch.Tensor:
    """Compute pairwise similarity matrix between features.
    
    Args:
        features1: First feature tensor of shape (n_samples, feature_dim)
        features2: Second feature tensor. If None, computes self-similarity of features1
        normalize: Whether to L2-normalize features
        metric: Similarity metric ('cosine' or 'euclidean')
        
    Returns:
        Similarity matrix of shape (n_samples1, n_samples2)
    """
    if features2 is None:
        features2 = features1
    
    if normalize:
        features1 = torch.nn.functional.normalize(features1, dim=1)
        features2 = torch.nn.functional.normalize(features2, dim=1)
    
    if metric == 'cosine':
        similarity = torch.matmul(features1, features2.T)
    elif metric == 'euclidean':
        # Negative euclidean distance (higher is more similar)
        similarity = -torch.cdist(features1, features2, p=2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return similarity


def compute_alignment_metrics(
    view1: torch.Tensor,
    view2: torch.Tensor,
    normalize: bool = True
) -> Dict[str, float]:
    """Compute various alignment metrics between two views.
    
    Args:
        view1: First view tensor of shape (batch_size, feature_dim)
        view2: Second view tensor of shape (batch_size, feature_dim)
        normalize: Whether to L2-normalize features
        
    Returns:
        Dictionary containing alignment metrics:
            - cosine_similarity: Mean cosine similarity
            - mse: Mean squared error
            - correlation: Mean correlation coefficient
    """
    if normalize:
        view1 = torch.nn.functional.normalize(view1, dim=1)
        view2 = torch.nn.functional.normalize(view2, dim=1)
    
    # Cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(view1, view2, dim=1)
    
    # MSE
    mse = torch.mean((view1 - view2) ** 2)
    
    # Correlation coefficient
    view1_centered = view1 - view1.mean(dim=0, keepdim=True)
    view2_centered = view2 - view2.mean(dim=0, keepdim=True)
    
    correlation = torch.sum(view1_centered * view2_centered, dim=1) / (
        torch.norm(view1_centered, dim=1) * torch.norm(view2_centered, dim=1) + 1e-8
    )
    
    return {
        'cosine_similarity': cosine_sim.mean().item(),
        'mse': mse.item(),
        'correlation': correlation.mean().item()
    }


def visualize_embeddings(
    embeddings: Union[torch.Tensor, np.ndarray],
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    method: str = 'tsne',
    title: str = 'Embedding Visualization',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """Visualize high-dimensional embeddings in 2D.
    
    Args:
        embeddings: Feature embeddings of shape (n_samples, feature_dim)
        labels: Optional labels for coloring points
        method: Dimensionality reduction method ('tsne' or 'pca')
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure. If None, doesn't save
        **kwargs: Additional arguments for reduction method
        
    Returns:
        Matplotlib figure object
    """
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Reduce to 2D
    if method.lower() == 'tsne':
        perplexity = kwargs.get('perplexity', 30)
        random_state = kwargs.get('random_state', 42)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    elif method.lower() == 'pca':
        random_state = kwargs.get('random_state', 42)
        reducer = PCA(n_components=2, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.6,
            s=50
        )
        plt.colorbar(scatter, ax=ax, label='Labels')
    else:
        ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.6,
            s=50
        )
    
    ax.set_title(title)
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_views(
    views: List[torch.Tensor],
    labels: Optional[torch.Tensor] = None,
    method: str = 'tsne',
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Visualize and compare multiple views side by side.
    
    Args:
        views: List of view tensors, each of shape (n_samples, feature_dim)
        labels: Optional labels for coloring points
        method: Dimensionality reduction method ('tsne' or 'pca')
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure object
    """
    num_views = len(views)
    fig, axes = plt.subplots(1, num_views, figsize=figsize)
    
    if num_views == 1:
        axes = [axes]
    
    for idx, (view, ax) in enumerate(zip(views, axes)):
        # Convert to numpy
        if isinstance(view, torch.Tensor):
            view_np = view.detach().cpu().numpy()
        else:
            view_np = view
        
        # Reduce to 2D
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2, random_state=42)
        
        view_2d = reducer.fit_transform(view_np)
        
        # Plot
        if labels is not None:
            labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            scatter = ax.scatter(
                view_2d[:, 0],
                view_2d[:, 1],
                c=labels_np,
                cmap='tab10',
                alpha=0.6,
                s=50
            )
            if idx == num_views - 1:
                plt.colorbar(scatter, ax=ax, label='Labels')
        else:
            ax.scatter(view_2d[:, 0], view_2d[:, 1], alpha=0.6, s=50)
        
        ax.set_title(f'View {idx + 1}')
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_simple_encoder(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = 'relu',
    dropout: float = 0.0,
    batch_norm: bool = False
) -> nn.Module:
    """Create a simple MLP encoder.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output feature dimension
        activation: Activation function name
        dropout: Dropout probability
        batch_norm: Whether to use batch normalization
        
    Returns:
        Sequential encoder model
    """
    layers = []
    dims = [input_dim] + hidden_dims + [output_dim]
    
    for i in range(len(dims) - 1):
        # Linear layer
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        
        # Batch norm (except last layer)
        if batch_norm and i < len(dims) - 2:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
        
        # Activation (except last layer)
        if i < len(dims) - 2:
            if activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'gelu':
                layers.append(nn.GELU())
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    
    return nn.Sequential(*layers)


def freeze_module(module: nn.Module) -> None:
    """Freeze all parameters in a module.
    
    Args:
        module: PyTorch module to freeze
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """Unfreeze all parameters in a module.
    
    Args:
        module: PyTorch module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """Count the number of parameters in a module.
    
    Args:
        module: PyTorch module
        trainable_only: If True, only count trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        PyTorch device object
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def batch_iterator(
    data: Union[torch.Tensor, np.ndarray],
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False
):
    """Create a simple batch iterator for data.
    
    Args:
        data: Data tensor or array
        batch_size: Size of each batch
        shuffle: Whether to shuffle data
        drop_last: Whether to drop the last incomplete batch
        
    Yields:
        Batches of data
    """
    n_samples = len(data)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        
        if drop_last and end_idx - start_idx < batch_size:
            break
        
        batch_indices = indices[start_idx:end_idx]
        
        if isinstance(data, torch.Tensor):
            yield data[batch_indices]
        else:
            yield data[batch_indices]


def compute_view_diversity(
    views: List[torch.Tensor],
    normalize: bool = True
) -> Dict[str, float]:
    """Compute diversity metrics between different views.
    
    Args:
        views: List of view tensors
        normalize: Whether to normalize features
        
    Returns:
        Dictionary with diversity metrics:
            - mean_pairwise_distance: Average distance between view pairs
            - min_similarity: Minimum similarity between any pair
            - max_similarity: Maximum similarity between any pair
    """
    num_views = len(views)
    
    if num_views < 2:
        return {'mean_pairwise_distance': 0.0, 'min_similarity': 1.0, 'max_similarity': 1.0}
    
    similarities = []
    
    for i in range(num_views):
        for j in range(i + 1, num_views):
            view1 = views[i]
            view2 = views[j]
            
            if normalize:
                view1 = torch.nn.functional.normalize(view1, dim=1)
                view2 = torch.nn.functional.normalize(view2, dim=1)
            
            # Compute mean cosine similarity
            sim = torch.nn.functional.cosine_similarity(view1, view2, dim=1).mean().item()
            similarities.append(sim)
    
    similarities = np.array(similarities)
    
    return {
        'mean_pairwise_distance': 1.0 - similarities.mean(),
        'min_similarity': similarities.min(),
        'max_similarity': similarities.max()
    }
