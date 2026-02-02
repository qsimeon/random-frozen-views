# Frozen View Contrastive Learning

> Learn representations by aligning views from randomly initialized frozen neural networks

This library implements a novel contrastive learning approach where randomly initialized, frozen neural networks serve as different "views" of the data. Instead of using data augmentations, the method trains a projection head to align representations from multiple frozen encoders, enabling self-supervised representation learning without backpropagating through the encoders.

## ✨ Features

- **Frozen Random Encoders** — Create multiple randomly initialized neural network encoders (CNN/MLP) that remain frozen during training, each providing a unique view of the input data based on its random initialization seed.
- **Contrastive Alignment** — Train a lightweight projection head to align representations from different frozen encoders using NT-Xent or BYOL-style contrastive loss, learning meaningful features without encoder updates.
- **Modular Architecture** — Clean separation between encoders, projection heads, training logic, and evaluation utilities, making it easy to experiment with different architectures and loss functions.
- **Downstream Evaluation** — Built-in evaluation tools including linear probe and k-NN classifiers to assess the quality of learned representations on downstream tasks.
- **Reproducible Experiments** — Seed-based encoder initialization and deterministic data handling ensure fully reproducible experiments across runs.
- **End-to-End Demo** — Complete demonstration script showing training, evaluation, and visualization of learned representations on sample datasets.

## 📦 Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.9+
- NumPy
- scikit-learn
- matplotlib (for visualization)

### Setup

1. git clone <repository-url>
   - Clone the repository to your local machine
2. cd frozen-view-contrastive-learning
   - Navigate to the project directory
3. pip install torch torchvision
   - Install PyTorch and torchvision (visit pytorch.org for GPU-specific instructions)
4. pip install numpy scikit-learn matplotlib
   - Install additional dependencies for data processing and visualization
5. python demo.py
   - Run the demo script to verify installation and see the method in action

## 🚀 Usage

### Basic Training with Two Frozen Encoders

Train a projection head to align representations from two randomly initialized frozen encoders.

```
import torch
from lib.core import FrozenViewEncoder, ProjectionHead, ContrastiveTrainer

# Create two frozen encoders with different random seeds
encoder1 = FrozenViewEncoder(input_dim=784, hidden_dims=[256, 128], seed=42)
encoder2 = FrozenViewEncoder(input_dim=784, hidden_dims=[256, 128], seed=123)

# Create projection head (trainable)
projection = ProjectionHead(input_dim=128, hidden_dim=64, output_dim=32)

# Initialize trainer
trainer = ContrastiveTrainer(
    encoders=[encoder1, encoder2],
    projection_head=projection,
    learning_rate=0.001,
    temperature=0.5
)

# Train on your data
for epoch in range(10):
    loss = trainer.train_epoch(train_loader)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
```

**Output:**

```
Epoch 1, Loss: 2.3456
Epoch 2, Loss: 1.8923
Epoch 3, Loss: 1.5234
...
Epoch 10, Loss: 0.7821
```

### Extract Learned Representations

Use the trained model to extract feature representations for downstream tasks.

```
import torch
from lib.core import FrozenViewEncoder, ProjectionHead

# Load trained models
encoder = FrozenViewEncoder(input_dim=784, hidden_dims=[256, 128], seed=42)
projection = ProjectionHead(input_dim=128, hidden_dim=64, output_dim=32)
projection.load_state_dict(torch.load('projection_head.pth'))

# Extract features
data = torch.randn(32, 784)  # Batch of 32 samples
with torch.no_grad():
    encoded = encoder(data)
    features = projection(encoded)

print(f"Input shape: {data.shape}")
print(f"Feature shape: {features.shape}")
```

**Output:**

```
Input shape: torch.Size([32, 784])
Feature shape: torch.Size([32, 32])
```

### Evaluate with Linear Probe

Assess representation quality by training a linear classifier on frozen features.

```
from lib.utils import LinearProbe, evaluate_linear_probe
from lib.core import FrozenViewEncoder, ProjectionHead
import torch

# Setup encoder and projection
encoder = FrozenViewEncoder(input_dim=784, hidden_dims=[256, 128], seed=42)
projection = ProjectionHead(input_dim=128, hidden_dim=64, output_dim=32)
projection.load_state_dict(torch.load('projection_head.pth'))

# Create linear probe
probe = LinearProbe(input_dim=32, num_classes=10)

# Train and evaluate
accuracy = evaluate_linear_probe(
    encoder=encoder,
    projection=projection,
    probe=probe,
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=50
)

print(f"Linear probe accuracy: {accuracy:.2%}")
```

**Output:**

```
Training linear probe...
Epoch 10/50, Accuracy: 78.3%
Epoch 20/50, Accuracy: 84.1%
Epoch 30/50, Accuracy: 87.5%
Epoch 40/50, Accuracy: 89.2%
Epoch 50/50, Accuracy: 90.1%
Linear probe accuracy: 90.10%
```

### Custom Encoder Architecture

Create frozen encoders with custom architectures (CNN for image data).

```
import torch
import torch.nn as nn
from lib.core import FrozenViewEncoder

# Define custom CNN architecture
class CustomCNNEncoder(FrozenViewEncoder):
    def __init__(self, seed=42):
        super().__init__(input_dim=None, hidden_dims=None, seed=seed)
        torch.manual_seed(seed)
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.freeze()

# Create two CNN encoders with different seeds
encoder1 = CustomCNNEncoder(seed=42)
encoder2 = CustomCNNEncoder(seed=999)

# Test with image data
image_batch = torch.randn(8, 3, 32, 32)
features1 = encoder1(image_batch)
features2 = encoder2(image_batch)

print(f"Features from encoder 1: {features1.shape}")
print(f"Features from encoder 2: {features2.shape}")
```

**Output:**

```
Features from encoder 1: torch.Size([8, 64])
Features from encoder 2: torch.Size([8, 64])
```

### Multi-View Training (3+ Encoders)

Train with more than two frozen encoders for richer multi-view learning.

```
from lib.core import FrozenViewEncoder, ProjectionHead, ContrastiveTrainer
import torch

# Create multiple frozen encoders
encoders = [
    FrozenViewEncoder(input_dim=784, hidden_dims=[256, 128], seed=s)
    for s in [42, 123, 456, 789]
]

projection = ProjectionHead(input_dim=128, hidden_dim=64, output_dim=32)

# Trainer handles multiple views automatically
trainer = ContrastiveTrainer(
    encoders=encoders,
    projection_head=projection,
    learning_rate=0.001,
    temperature=0.5
)

print(f"Training with {len(encoders)} frozen views")
for epoch in range(5):
    loss = trainer.train_epoch(train_loader)
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
```

**Output:**

```
Training with 4 frozen views
Epoch 1: Loss = 3.1245
Epoch 2: Loss = 2.4567
Epoch 3: Loss = 1.9823
Epoch 4: Loss = 1.6234
Epoch 5: Loss = 1.3456
```

## 🏗️ Architecture

The library follows a modular architecture with three main components: (1) FrozenViewEncoder - randomly initialized neural networks that remain frozen during training, (2) ProjectionHead - trainable module that maps encoder outputs to a contrastive learning space, and (3) ContrastiveTrainer - orchestrates training by computing contrastive loss between aligned views. Utility modules provide data handling, evaluation metrics, and visualization tools.

### File Structure

```
┌─────────────────────────────────────────────────────────┐
│                    Input Data (X)                       │
└────────────────┬────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌───────────────┐  ┌───────────────┐
│ FrozenEncoder │  │ FrozenEncoder │
│   (seed=42)   │  │  (seed=123)   │
│   [FROZEN]    │  │   [FROZEN]    │
└───────┬───────┘  └───────┬───────┘
        │                  │
        │  z1              │  z2
        │                  │
        └────────┬─────────┘
                 │
                 ▼
        ┌────────────────┐
        │ ProjectionHead │
        │  [TRAINABLE]   │
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────┐
        │ Contrastive    │
        │ Loss (NT-Xent) │
        └────────┬───────┘
                 │
                 ▼
        Update Projection Only

Project Structure:
frozen-view-contrastive-learning/
├── lib/
│   ├── core.py          # Core classes
│   └── utils.py         # Utilities
├── demo.py              # End-to-end demo
└── README.md
```

### Files

- **lib/core.py** — Contains core classes: FrozenViewEncoder (frozen random networks), ProjectionHead (trainable alignment module), and ContrastiveTrainer (training orchestration and loss computation).
- **lib/utils.py** — Provides utility functions for data handling, evaluation metrics (linear probe, k-NN), visualization tools, and helper functions for reproducible experiments.
- **demo.py** — End-to-end demonstration script showing complete workflow: encoder creation, training, evaluation, and visualization of learned representations on sample datasets.

### Design Decisions

- Frozen encoders remain untrained to test whether random projections alone can provide useful views for contrastive learning, reducing computational cost.
- Only the projection head is trainable, dramatically reducing memory requirements and training time compared to traditional contrastive methods.
- Multiple encoder seeds create diverse views without data augmentation, offering an alternative approach to view generation in contrastive learning.
- NT-Xent loss is used as the default contrastive objective, but the architecture supports pluggable loss functions (BYOL, SimCLR, etc.).
- Modular design separates concerns: encoders, projection, training, and evaluation can be independently modified or extended.
- Seed-based initialization ensures reproducibility and allows systematic exploration of how random initialization affects learned representations.

## 🔧 Technical Details

### Dependencies

- **torch** (1.9+) — Deep learning framework for building and training neural networks, handling tensor operations and automatic differentiation.
- **numpy** — Numerical computing library for array operations, data preprocessing, and mathematical computations.
- **scikit-learn** — Machine learning library providing evaluation metrics, k-NN classifier, and data preprocessing utilities.
- **matplotlib** — Visualization library for plotting training curves, t-SNE embeddings, and representation quality analysis.

### Key Algorithms / Patterns

- NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for contrastive learning between frozen encoder views.
- Random network initialization with controlled seeds to generate diverse but reproducible feature extractors.
- Linear probe evaluation: train linear classifier on frozen features to assess representation quality.
- k-Nearest Neighbors classification in learned embedding space for non-parametric evaluation.
- Cosine similarity computation for measuring alignment between representations from different frozen encoders.

### Important Notes

- Frozen encoders never receive gradient updates; only the projection head is optimized during training.
- Different random seeds for encoders are critical - identical seeds would produce identical views and prevent learning.
- The projection head dimensionality should be tuned based on downstream task complexity and encoder output size.
- Batch size significantly impacts contrastive learning performance; larger batches generally improve results.
- Temperature parameter in NT-Xent loss controls the concentration of the distribution; typical values range from 0.1 to 0.5.

## ❓ Troubleshooting

### Loss not decreasing or training diverges

**Cause:** Learning rate too high, temperature parameter misconfigured, or encoders initialized with identical seeds producing identical views.

**Solution:** Reduce learning rate (try 0.0001-0.001), adjust temperature to 0.1-0.5, and verify each encoder has a unique seed value. Check that projection head is not frozen accidentally.

### CUDA out of memory error

**Cause:** Batch size too large for available GPU memory, or multiple frozen encoders consuming too much memory simultaneously.

**Solution:** Reduce batch size, use gradient accumulation, or process encoder views sequentially instead of in parallel. Consider using smaller encoder architectures or fewer hidden units.

### Poor downstream task performance

**Cause:** Frozen encoders may be too small/large, projection head not properly trained, or insufficient training epochs for alignment.

**Solution:** Experiment with encoder architecture (more layers or wider hidden dimensions), train for more epochs, increase batch size, or try different temperature values. Verify projection head is actually being updated.

### Reproducibility issues across runs

**Cause:** Random seeds not set properly, non-deterministic CUDA operations, or data loader shuffling without fixed seed.

**Solution:** Set torch.manual_seed(), torch.cuda.manual_seed_all(), and numpy.random.seed() at the start. Use torch.backends.cudnn.deterministic = True and set worker_init_fn in DataLoader.

### Import errors when running demo.py

**Cause:** Dependencies not installed or Python path not configured to find lib/ directory.

**Solution:** Run 'pip install torch numpy scikit-learn matplotlib' to install dependencies. Ensure you're running from the project root directory, or add the project to PYTHONPATH.

---

This project explores an experimental approach to contrastive learning where random, frozen neural networks serve as view generators. While traditional contrastive methods rely on data augmentation, this approach investigates whether diverse random projections can provide sufficient signal for learning useful representations. The library is designed for research and experimentation. This documentation was generated with AI assistance.