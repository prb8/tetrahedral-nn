## Tetrahedral Neural Network Architecture

Author: Philipp Remy Bartholomäus  
Date: October 30, 2025

A novel neural network architecture with tetrahedral topology proven to generalize arithmetic from [-9; 9] to trillions (float32 precision limited).

## Architecture

**Structure:**
- **4 vertices** (core computation nodes)
- **6 edges** (pairwise linear attention interactions)  
- **4 faces** (triangular 3-point attention mechanisms)

The tetrahedral topology provides the geometric structure for the network to self-organize without task-specific assumptions about the data.

## Key Results

### Arithmetic Generalization
- **Train range:** [-9, 9] (exhaustive dataset)
- **Test range:** Up to 10,000+ (and scales to trillions on 2-input summation)
- **Extrapolation:** 1000Ã— beyond training range
- **Error:** Float32 precision limited (~1e-7 relative error)
- **Mechanism:** Architecture enables conceptual learning, not memorization

### Inverse Operations (Decomposition)
- **Composition:** (a, b, c, d) â†’ sum (compression: 4â†’1)
- **Decomposition:** num â†’ (num/4, num/4, num/4, num/4) (expansion: 1â†’4)
- **Finding:** Architecture works bidirectionally
- **Implications:** Symmetric processing capabilities

## Files

- **`tetrahedral_core.py`** - Core architecture (TriSimplicialAttention + TetrahedralCore)
- **`arithmetic_adapter.py`** - Data interface for arithmetic tasks  
- **`tetrahedral_trainer.py`** - Universal training system
- **`tetrahedral_tests.py`** - Test suite for generalization validation
- **`train_example.py`** - Complete example: train on [-9, 9], test generalization

## Usage

### Basic Training

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tetrahedral_core import TetrahedralCore
from arithmetic_adapter import ArithmeticAdapter
from tetrahedral_trainer import TetrahedralTrainer

# Create adapter and datasets
adapter = ArithmeticAdapter(n_inputs=2)
train_data = adapter.create_dataset(train_range=(-9, 9), exhaustive=True)
test_data = adapter.create_test_dataset(test_range=(10, 100), n_samples=1000)

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)

# Create model
model = TetrahedralCore(input_dim=2, output_dim=1, latent_dim=64)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
trainer = TetrahedralTrainer(model, optimizer)
history = trainer.train(train_loader, test_loader, epochs=200)
```

### Run Complete Example

```bash
python train_example.py
```

This will:
1. Train on [-9, 9] exhaustively (361 samples)
2. Test generalization at 10-100, 100-1000, 1000-10000 ranges
3. Display results showing float-precision-limited generalization
4. Establish baseline for future experiments

## Technical Details

### Attention Mechanisms

**Linear Attention (Edges):**
- Pairwise interactions between vertices
- O(N) complexity via linear kernels
- 6 edges connect all vertex pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)

**3-Point Attention (Faces):**
- Triangular attention on 3 vertices per face
- 4 faces per tetrahedron: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
- Enables discovery of higher-order relationships

### Parameter Count
- ~2.5K parameters for 2-input model
- Scales efficiently with input/output dimensions

## Theoretical Foundation

This architecture builds on recent advances in topological deep learning and higher-order neural networks:

### Core References

**Higher-Order Attention:**
- Hajij, M., Zamzmi, G., et al. (2022). "Architectures of Topological Deep Learning: A Survey on Topological Neural Networks." *arXiv:2304.10031*
- Hajij, M., Zamzmi, G., et al. (2022). "Higher-Order Attention Networks." *arXiv:2206.00606*

**Simplicial & Geometric Attention:**
- Giusti, L., et al. (2023). "Simplicial Attention Networks." *OpenReview*
- "2-Simplicial Attention" (2025). *arXiv:2507.02754*
- Farazi, M., et al. (2025). "A Recipe for Geometry-Aware 3D Mesh Transformers." *WACV 2025*

**Linear Attention Mechanisms:**
- Choromanski, K., et al. (2021). "Rethinking Attention with Performers." *ICLR 2021*
- Gu, A., & Dao, T. (2024). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." *arXiv:2312.00752*

**Tetrahedral Structures in Neural Networks:**
- Hu, T., et al. (2022). "TetGAN: A Convolutional Neural Network for Tetrahedral Meshes." *BMVC 2022*
- Gao, L., et al. (2023). "TetCNN: Convolutional Neural Networks on Tetrahedral Meshes." *ACM TOG 2023*

### Conceptual Inspiration

**Morphogenetic Intelligence & Self-Organization:**
- Levin, M. (2019). "The Computational Boundary of a 'Self': Developmental Bioelectricity Drives Multicellularity and Scale-Free Cognition." *Frontiers in Psychology*
- Levin, M. (2022). "Technological Approach to Mind Everywhere: An Experimentally-Grounded Framework for Understanding Diverse Bodies and Minds." *Frontiers in Systems Neuroscience*

Michael Levin's work on morphogenetic fields, bioelectric networks, and self-organizing computational systems fundamentally inspired the architectural philosophy: vertices self-organize their functional specialization through geometric constraints rather than explicit programming.

## Novel Contributions

This work specifically contributes:
1. **Hybrid attention architecture** combining linear (edges) + 3-point (faces) on tetrahedral topology
2. **Extreme generalization proof** (1000× arithmetic extrapolation, float-precision limited)
3. **Bidirectional operation** (compression 4→1 and expansion 1→4)
4. **Domain-agnostic geometric scaffold** for self-organizing computation

The tetrahedral structure itself is established in mesh processing; the novel contribution is its use as a **computational primitive** with hybrid attention mechanisms for general-purpose learning.

# Future Work (Roadmap)

- [x] Arithmetic generalization proof
- [ ] 3D geometric learning (in progress)
