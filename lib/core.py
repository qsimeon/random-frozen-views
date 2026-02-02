"""Core module for contrastive learning with frozen random neural network views.

This module provides the main components for using randomly initialized frozen
neural network layers as different "views" of the data for contrastive learning
and alignment. The frozen random projections serve as data augmentation alternatives
that preserve semantic information while providing diverse representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Callable
import numpy as np


class FrozenRandomProjection(nn.Module):
    """A frozen randomly initialized neural network layer for data projection.
    
    This layer creates a random projection of input data that remains fixed
    during training. It serves as a deterministic "view" generator for
    contrastive learning.
    
    Args:
        input_dim: Dimension of input features
        output_dim: Dimension of output features
        activation: Activation function to use ('relu', 'tanh', 'sigmoid', or None)
        bias: Whether to include bias term
        init_scale: Scale factor for weight initialization
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Optional[str] = 'relu',
        bias: bool = True,
        init_scale: float = 1.0,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
        
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        
        # Initialize weights
        nn.init.xavier_normal_(self.linear.weight, gain=init_scale)
        if bias:
            nn.init.zeros_(self.linear.bias)
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Set activation
        self.activation = self._get_activation(activation)
        
    def _get_activation(self, activation: Optional[str]) -> Optional[Callable]:
        """Get activation function from string name."""
        if activation is None or activation.lower() == 'none':
            return None
        elif activation.lower() == 'relu':
            return F.relu
        elif activation.lower() == 'tanh':
            return torch.tanh
        elif activation.lower() == 'sigmoid':
            return torch.sigmoid
        elif activation.lower() == 'gelu':
            return F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through frozen projection.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Projected tensor of shape (batch_size, output_dim)
        """
        out = self.linear(x)
        if self.activation is not None:
            out = self.activation(out)
        return out


class MultiViewGenerator(nn.Module):
    """Generate multiple views of data using frozen random projections.
    
    Creates multiple independent frozen random neural network layers,
    each providing a different "view" of the input data for contrastive learning.
    
    Args:
        input_dim: Dimension of input features
        projection_dim: Dimension of each projection output
        num_views: Number of different views to generate
        hidden_dims: List of hidden layer dimensions for each projection
        activation: Activation function for hidden layers
        seed_offset: Starting seed for view generators (each view gets seed_offset + i)
    """
    
    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        num_views: int = 2,
        hidden_dims: Optional[List[int]] = None,
        activation: str = 'relu',
        seed_offset: int = 42
    ):
        super().__init__()
        
        self.num_views = num_views
        self.projection_dim = projection_dim
        
        # Create multiple frozen projection networks
        self.projections = nn.ModuleList()
        
        for i in range(num_views):
            if hidden_dims is None or len(hidden_dims) == 0:
                # Single layer projection
                proj = FrozenRandomProjection(
                    input_dim=input_dim,
                    output_dim=projection_dim,
                    activation=activation,
                    seed=seed_offset + i
                )
            else:
                # Multi-layer projection
                layers = []
                dims = [input_dim] + hidden_dims + [projection_dim]
                
                for j in range(len(dims) - 1):
                    is_last = (j == len(dims) - 2)
                    act = None if is_last else activation
                    
                    layer = FrozenRandomProjection(
                        input_dim=dims[j],
                        output_dim=dims[j + 1],
                        activation=act,
                        seed=seed_offset + i * 100 + j
                    )
                    layers.append(layer)
                
                proj = nn.Sequential(*layers)
            
            self.projections.append(proj)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Generate multiple views of input data.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            List of projected tensors, each of shape (batch_size, projection_dim)
        """
        return [proj(x) for proj in self.projections]
    
    def get_view(self, x: torch.Tensor, view_idx: int) -> torch.Tensor:
        """Get a specific view of the input data.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            view_idx: Index of the view to generate
            
        Returns:
            Projected tensor of shape (batch_size, projection_dim)
        """
        if view_idx >= self.num_views:
            raise ValueError(f"View index {view_idx} out of range (num_views={self.num_views})")
        return self.projections[view_idx](x)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for aligning multiple views.
    
    Implements NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
    commonly used in contrastive learning frameworks like SimCLR.
    
    Args:
        temperature: Temperature parameter for scaling
        reduction: Reduction method ('mean' or 'sum')
    """
    
    def __init__(self, temperature: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """Compute contrastive loss between two views.
        
        Args:
            view1: First view tensor of shape (batch_size, feature_dim)
            view2: Second view tensor of shape (batch_size, feature_dim)
            normalize: Whether to L2-normalize features before computing similarity
            
        Returns:
            Scalar loss tensor
        """
        batch_size = view1.shape[0]
        
        # Normalize features
        if normalize:
            view1 = F.normalize(view1, dim=1)
            view2 = F.normalize(view2, dim=1)
        
        # Concatenate views
        features = torch.cat([view1, view2], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels: positive pairs are (i, i+batch_size) and (i+batch_size, i)
        labels = torch.arange(batch_size, device=view1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Mask to remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=view1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            similarity_matrix,
            labels,
            reduction=self.reduction
        )
        
        return loss


class AlignmentLoss(nn.Module):
    """Alignment loss for matching representations across views.
    
    Computes the mean squared error or cosine distance between aligned views.
    
    Args:
        loss_type: Type of alignment loss ('mse' or 'cosine')
        reduction: Reduction method ('mean' or 'sum')
    """
    
    def __init__(self, loss_type: str = 'mse', reduction: str = 'mean'):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.reduction = reduction
        
        if self.loss_type not in ['mse', 'cosine']:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """Compute alignment loss between two views.
        
        Args:
            view1: First view tensor of shape (batch_size, feature_dim)
            view2: Second view tensor of shape (batch_size, feature_dim)
            normalize: Whether to L2-normalize features
            
        Returns:
            Scalar loss tensor
        """
        if normalize:
            view1 = F.normalize(view1, dim=1)
            view2 = F.normalize(view2, dim=1)
        
        if self.loss_type == 'mse':
            loss = F.mse_loss(view1, view2, reduction=self.reduction)
        else:  # cosine
            # Cosine distance = 1 - cosine_similarity
            cosine_sim = F.cosine_similarity(view1, view2, dim=1)
            loss = 1 - cosine_sim
            
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
        
        return loss


class FrozenViewContrastiveModel(nn.Module):
    """Complete model for contrastive learning with frozen random views.
    
    Combines a learnable encoder with frozen random projection views and
    contrastive loss for self-supervised representation learning.
    
    Args:
        encoder: Learnable encoder network
        input_dim: Dimension of encoder output
        projection_dim: Dimension of frozen projections
        num_views: Number of frozen views to generate
        hidden_dims: Hidden dimensions for frozen projections
        temperature: Temperature for contrastive loss
        activation: Activation function for frozen layers
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        projection_dim: int = 128,
        num_views: int = 2,
        hidden_dims: Optional[List[int]] = None,
        temperature: float = 0.5,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.encoder = encoder
        self.view_generator = MultiViewGenerator(
            input_dim=input_dim,
            projection_dim=projection_dim,
            num_views=num_views,
            hidden_dims=hidden_dims,
            activation=activation
        )
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
    
    def forward(
        self,
        x: torch.Tensor,
        return_views: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass through encoder and view generation.
        
        Args:
            x: Input tensor
            return_views: Whether to return the generated views
            
        Returns:
            If return_views is False: encoded features
            If return_views is True: tuple of (encoded features, list of views)
        """
        # Encode input
        features = self.encoder(x)
        
        if return_views:
            # Generate views
            views = self.view_generator(features)
            return features, views
        else:
            return features
    
    def compute_loss(
        self,
        x: torch.Tensor,
        view_pairs: Optional[List[Tuple[int, int]]] = None
    ) -> torch.Tensor:
        """Compute contrastive loss for the input.
        
        Args:
            x: Input tensor
            view_pairs: List of (view_idx1, view_idx2) pairs to contrast.
                       If None, uses all consecutive pairs.
            
        Returns:
            Scalar loss tensor
        """
        features, views = self.forward(x, return_views=True)
        
        if view_pairs is None:
            # Default: contrast consecutive view pairs
            view_pairs = [(i, i + 1) for i in range(len(views) - 1)]
        
        total_loss = 0.0
        for idx1, idx2 in view_pairs:
            loss = self.contrastive_loss(views[idx1], views[idx2])
            total_loss += loss
        
        return total_loss / len(view_pairs)
