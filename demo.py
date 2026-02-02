"""
Frozen Random Neural Network Views for Contrastive Learning Demo

This demo demonstrates how to use randomly initialized frozen neural network layers
as different "views" of the data for contrastive learning and alignment.

The key idea is that frozen random projections can serve as diverse feature extractors
without requiring training, and we can learn to align representations across these views.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import sys
import os

# Add lib directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# Import from our modules
from core import (
    FrozenRandomProjection,
    MultiViewGenerator,
    ContrastiveLoss,
    AlignmentLoss,
    FrozenViewContrastiveModel
)
from utils import (
    set_seed,
    compute_similarity_matrix,
    compute_alignment_metrics,
    visualize_embeddings,
    compare_views,
    create_simple_encoder,
    freeze_module,
    count_parameters,
    get_device,
    batch_iterator,
    compute_view_diversity
)


def generate_synthetic_data(n_samples: int = 1000, n_features: int = 50, n_classes: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        n_classes: Number of classes
        
    Returns:
        Tuple of (data, labels)
    """
    # Generate class centers
    centers = torch.randn(n_classes, n_features) * 3
    
    # Generate samples around centers
    samples_per_class = n_samples // n_classes
    data_list = []
    labels_list = []
    
    for i in range(n_classes):
        # Generate samples with some noise around each center
        class_data = centers[i].unsqueeze(0) + torch.randn(samples_per_class, n_features) * 0.5
        data_list.append(class_data)
        labels_list.append(torch.full((samples_per_class,), i, dtype=torch.long))
    
    data = torch.cat(data_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    # Shuffle
    perm = torch.randperm(data.size(0))
    data = data[perm]
    labels = labels[perm]
    
    return data, labels


def demo_frozen_random_projections():
    """
    Demo 1: Show how frozen random projections create different views of data.
    """
    print("=" * 80)
    print("DEMO 1: Frozen Random Projections as Data Views")
    print("=" * 80)
    
    set_seed(42)
    device = get_device()
    
    # Generate synthetic data
    data, labels = generate_synthetic_data(n_samples=500, n_features=50, n_classes=5)
    data = data.to(device)
    labels = labels.to(device)
    
    print(f"\nGenerated data shape: {data.shape}")
    print(f"Number of classes: {len(torch.unique(labels))}")
    
    # Create multiple frozen random projections
    n_views = 3
    projections = []
    
    for i in range(n_views):
        proj = FrozenRandomProjection(
            input_dim=50,
            output_dim=32,
            hidden_dims=[64, 48],
            activation='relu'
        ).to(device)
        projections.append(proj)
        
        # Verify it's frozen
        trainable_params = count_parameters(proj, trainable_only=True)
        total_params = count_parameters(proj, trainable_only=False)
        print(f"\nView {i+1}: {total_params} total params, {trainable_params} trainable (frozen: {trainable_params == 0})")
    
    # Generate views
    views = [proj(data) for proj in projections]
    
    # Compute diversity between views
    print("\n" + "-" * 80)
    print("View Diversity Analysis:")
    print("-" * 80)
    diversity_metrics = compute_view_diversity(views, normalize=True)
    for metric, value in diversity_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize views
    print("\nGenerating visualization of different views...")
    try:
        fig = compare_views(
            views=[v.detach().cpu() for v in views],
            labels=labels.cpu(),
            method='tsne',
            figsize=(15, 5)
        )
        plt.savefig('frozen_views_comparison.png', dpi=150, bbox_inches='tight')
        print("✓ Saved visualization to 'frozen_views_comparison.png'")
        plt.close()
    except Exception as e:
        print(f"⚠ Visualization skipped: {e}")
    
    return data, labels, views


def demo_multi_view_generator():
    """
    Demo 2: Use MultiViewGenerator to create multiple views efficiently.
    """
    print("\n" + "=" * 80)
    print("DEMO 2: Multi-View Generator")
    print("=" * 80)
    
    set_seed(42)
    device = get_device()
    
    # Generate data
    data, labels = generate_synthetic_data(n_samples=500, n_features=50, n_classes=5)
    data = data.to(device)
    
    # Create multi-view generator
    view_generator = MultiViewGenerator(
        input_dim=50,
        output_dim=32,
        n_views=4,
        hidden_dims=[64, 48],
        activation='relu'
    ).to(device)
    
    print(f"\nCreated MultiViewGenerator with {count_parameters(view_generator)} parameters")
    print(f"Trainable parameters: {count_parameters(view_generator, trainable_only=True)}")
    
    # Generate all views at once
    all_views = view_generator(data)
    print(f"\nGenerated {len(all_views)} views")
    for i, view in enumerate(all_views):
        print(f"View {i+1} shape: {view.shape}")
    
    # Get individual views
    print("\nAccessing individual views:")
    view_0 = view_generator.get_view(data, view_idx=0)
    view_2 = view_generator.get_view(data, view_idx=2)
    print(f"View 0 shape: {view_0.shape}")
    print(f"View 2 shape: {view_2.shape}")
    
    # Compute alignment between views
    print("\n" + "-" * 80)
    print("Alignment Metrics Between Views:")
    print("-" * 80)
    
    for i in range(len(all_views) - 1):
        metrics = compute_alignment_metrics(
            all_views[i].detach(),
            all_views[i + 1].detach(),
            normalize=True
        )
        print(f"\nView {i+1} vs View {i+2}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return view_generator, data, labels


def demo_contrastive_learning():
    """
    Demo 3: Train a model using contrastive learning across frozen views.
    """
    print("\n" + "=" * 80)
    print("DEMO 3: Contrastive Learning with Frozen Views")
    print("=" * 80)
    
    set_seed(42)
    device = get_device()
    
    # Generate data
    data, labels = generate_synthetic_data(n_samples=1000, n_features=50, n_classes=5)
    data = data.to(device)
    labels = labels.to(device)
    
    # Split into train and test
    n_train = 800
    train_data, test_data = data[:n_train], data[n_train:]
    train_labels, test_labels = labels[:n_train], labels[n_train:]
    
    print(f"\nTrain data: {train_data.shape}")
    print(f"Test data: {test_data.shape}")
    
    # Create the contrastive model
    model = FrozenViewContrastiveModel(
        input_dim=50,
        view_output_dim=32,
        encoder_hidden_dims=[64, 48],
        projection_dim=16,
        n_views=3,
        view_hidden_dims=[64, 48],
        temperature=0.5
    ).to(device)
    
    print(f"\nModel created with {count_parameters(model)} total parameters")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True)}")
    
    # Setup optimizer (only trainable parameters)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    n_epochs = 50
    batch_size = 64
    
    print("\n" + "-" * 80)
    print("Training Progress:")
    print("-" * 80)
    
    train_losses = []
    
    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        
        for batch_data in batch_iterator(train_data, batch_size=batch_size, shuffle=True):
            batch_data = batch_data.to(device)
            
            # Forward pass
            embeddings, views = model(batch_data)
            
            # Compute loss
            loss = model.compute_loss(embeddings, views)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")
    
    # Evaluate on test data
    print("\n" + "-" * 80)
    print("Evaluation on Test Data:")
    print("-" * 80)
    
    model.eval()
    with torch.no_grad():
        # Get embeddings and views for test data
        test_embeddings, test_views = model(test_data)
        
        # Compute alignment metrics between learned embeddings and frozen views
        print("\nAlignment between learned embeddings and frozen views:")
        for i, view in enumerate(test_views):
            metrics = compute_alignment_metrics(
                test_embeddings.detach(),
                view.detach(),
                normalize=True
            )
            print(f"\nEmbedding vs View {i+1}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Visualize learned embeddings
        print("\nGenerating embedding visualization...")
        try:
            fig = visualize_embeddings(
                test_embeddings.cpu().numpy(),
                labels=test_labels.cpu().numpy(),
                method='tsne',
                title='Learned Embeddings (Contrastive Learning)',
                figsize=(10, 8)
            )
            plt.savefig('learned_embeddings.png', dpi=150, bbox_inches='tight')
            print("✓ Saved visualization to 'learned_embeddings.png'")
            plt.close()
        except Exception as e:
            print(f"⚠ Visualization skipped: {e}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Contrastive Loss', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    print("✓ Saved training loss plot to 'training_loss.png'")
    plt.close()
    
    return model, train_losses


def demo_alignment_loss():
    """
    Demo 4: Demonstrate different loss functions for alignment.
    """
    print("\n" + "=" * 80)
    print("DEMO 4: Alignment Loss Functions")
    print("=" * 80)
    
    set_seed(42)
    device = get_device()
    
    # Generate data
    data, _ = generate_synthetic_data(n_samples=200, n_features=50, n_classes=5)
    data = data.to(device)
    
    # Create views
    view_generator = MultiViewGenerator(
        input_dim=50,
        output_dim=32,
        n_views=2,
        hidden_dims=[64]
    ).to(device)
    
    views = view_generator(data)
    view1, view2 = views[0], views[1]
    
    # Test ContrastiveLoss
    print("\nContrastive Loss:")
    contrastive_loss_fn = ContrastiveLoss(temperature=0.5)
    contrastive_loss = contrastive_loss_fn(view1, view2)
    print(f"Loss value: {contrastive_loss.item():.4f}")
    
    # Test AlignmentLoss with different modes
    print("\nAlignment Loss (MSE mode):")
    alignment_loss_mse = AlignmentLoss(mode='mse')
    loss_mse = alignment_loss_mse(view1, view2)
    print(f"Loss value: {loss_mse.item():.4f}")
    
    print("\nAlignment Loss (Cosine mode):")
    alignment_loss_cosine = AlignmentLoss(mode='cosine')
    loss_cosine = alignment_loss_cosine(view1, view2)
    print(f"Loss value: {loss_cosine.item():.4f}")
    
    # Compare similarity matrices
    print("\n" + "-" * 80)
    print("Similarity Analysis:")
    print("-" * 80)
    
    sim_matrix = compute_similarity_matrix(view1, view2, normalize=True, metric='cosine')
    print(f"\nSimilarity matrix shape: {sim_matrix.shape}")
    print(f"Mean similarity: {sim_matrix.mean().item():.4f}")
    print(f"Std similarity: {sim_matrix.std().item():.4f}")
    
    # Visualize similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_matrix.detach().cpu().numpy()[:50, :50], cmap='viridis', aspect='auto')
    plt.colorbar(label='Cosine Similarity')
    plt.xlabel('View 2 Samples', fontsize=12)
    plt.ylabel('View 1 Samples', fontsize=12)
    plt.title('Cross-View Similarity Matrix (First 50 samples)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('similarity_matrix.png', dpi=150, bbox_inches='tight')
    print("✓ Saved similarity matrix to 'similarity_matrix.png'")
    plt.close()


def demo_custom_encoder_with_frozen_views():
    """
    Demo 5: Create a custom encoder and align it with frozen views.
    """
    print("\n" + "=" * 80)
    print("DEMO 5: Custom Encoder with Frozen View Alignment")
    print("=" * 80)
    
    set_seed(42)
    device = get_device()
    
    # Generate data
    data, labels = generate_synthetic_data(n_samples=800, n_features=50, n_classes=5)
    data = data.to(device)
    labels = labels.to(device)
    
    # Create frozen views
    frozen_view = FrozenRandomProjection(
        input_dim=50,
        output_dim=32,
        hidden_dims=[64, 48],
        activation='relu'
    ).to(device)
    
    print(f"Frozen view parameters: {count_parameters(frozen_view)}")
    print(f"Frozen view trainable: {count_parameters(frozen_view, trainable_only=True)}")
    
    # Create trainable encoder
    encoder = create_simple_encoder(
        input_dim=50,
        hidden_dims=[64, 48],
        output_dim=32,
        activation='relu',
        dropout=0.1,
        batch_norm=True
    ).to(device)
    
    print(f"\nEncoder parameters: {count_parameters(encoder)}")
    print(f"Encoder trainable: {count_parameters(encoder, trainable_only=True)}")
    
    # Setup training
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    loss_fn = AlignmentLoss(mode='cosine')
    
    # Training loop
    n_epochs = 100
    batch_size = 64
    losses = []
    
    print("\n" + "-" * 80)
    print("Training Custom Encoder:")
    print("-" * 80)
    
    for epoch in range(n_epochs):
        encoder.train()
        epoch_losses = []
        
        for batch_data in batch_iterator(data, batch_size=batch_size, shuffle=True):
            batch_data = batch_data.to(device)
            
            # Get frozen view (no gradients needed)
            with torch.no_grad():
                frozen_features = frozen_view(batch_data)
            
            # Get encoder output
            encoder_features = encoder(batch_data)
            
            # Compute alignment loss
            loss = loss_fn(encoder_features, frozen_features)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")
    
    # Evaluate alignment
    print("\n" + "-" * 80)
    print("Final Alignment Metrics:")
    print("-" * 80)
    
    encoder.eval()
    with torch.no_grad():
        frozen_features = frozen_view(data)
        encoder_features = encoder(data)
        
        metrics = compute_alignment_metrics(
            encoder_features,
            frozen_features,
            normalize=True
        )
        
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2, color='steelblue')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Alignment Loss', fontsize=12)
    plt.title('Custom Encoder Training: Alignment with Frozen View', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('custom_encoder_training.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved training plot to 'custom_encoder_training.png'")
    plt.close()


def main():
    """
    Main function to run all demos.
    """
    print("\n" + "=" * 80)
    print("FROZEN RANDOM NEURAL NETWORK VIEWS FOR CONTRASTIVE LEARNING")
    print("=" * 80)
    print("\nThis demo showcases how randomly initialized frozen neural networks")
    print("can serve as different 'views' of data for contrastive learning.")
    print("\nKey concepts:")
    print("  • Frozen random projections create diverse feature spaces")
    print("  • No training required for the view generators")
    print("  • Learn to align representations across these fixed views")
    print("  • Useful for self-supervised learning and representation learning")
    
    try:
        # Run all demos
        demo_frozen_random_projections()
        demo_multi_view_generator()
        demo_contrastive_learning()
        demo_alignment_loss()
        demo_custom_encoder_with_frozen_views()
        
        print("\n" + "=" * 80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nGenerated files:")
        print("  • frozen_views_comparison.png - Visualization of different frozen views")
        print("  • learned_embeddings.png - Learned embeddings from contrastive learning")
        print("  • training_loss.png - Contrastive learning training loss")
        print("  • similarity_matrix.png - Cross-view similarity matrix")
        print("  • custom_encoder_training.png - Custom encoder alignment training")
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during demo execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
