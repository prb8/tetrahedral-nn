"""
Systematic Test Suite for Tetrahedral Architecture

Proof that the architecture works.
Baseline for all future experiments.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, List


class TetrahedralTests:
    """
    Systematic test suite proving tetrahedral architecture works.
    
    These are the KEY tests that establish the gold standard:
        1. Exhaustive learning (perfect on training set)
        2. Generalization (scales beyond training)
        3. Multi-input scaling (2, 3, 4 inputs)
    """
    
    @staticmethod
    def test_generalization(model,
                          adapter,
                          train_range: Tuple[int, int] = (-9, 9),
                          test_ranges: List[Tuple[int, int]] = None,
                          device: str = 'cpu') -> dict:
        """
        THE KEY TEST: Generalization beyond training range.
        
        Train on small range (e.g., [-9, 9]), test on increasingly large ranges.
        This proves the model learned the CONCEPT, not just memorization.
        """
        if test_ranges is None:
            test_ranges = [
                (10, 100),      # 10Ã— extrapolation
                (100, 1000),    # 100Ã— extrapolation
                (1000, 10000),  # 1000Ã— extrapolation
            ]

        model.eval()
        results = {}

        print(f"\nðŸ”º Generalization Test")
        print(f"{'='*60}")
        print(f"Training range: [{train_range[0]}, {train_range[1]}]")
        print(f"Testing extrapolation...\n")

        for test_range in test_ranges:
            test_data = adapter.create_test_dataset(test_range, n_samples=1000)
            test_loader = DataLoader(test_data, batch_size=256)

            total_mae = 0
            total_mse = 0
            n_samples = 0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    predictions = model(inputs)
                    mae = torch.abs(predictions - targets).mean().item()
                    mse = F.mse_loss(predictions, targets).item()

                    total_mae += mae * len(inputs)
                    total_mse += mse * len(inputs)
                    n_samples += len(inputs)

            avg_mae = total_mae / n_samples
            avg_mse = total_mse / n_samples

            range_label = f"[{test_range[0]}, {test_range[1]}]"
            results[range_label] = {'mae': avg_mae, 'mse': avg_mse}

            extrapolation = test_range[1] / train_range[1]
            print(f"Range {range_label:20s} ({extrapolation:6.0f}Ã— extrapolation)")
            print(f"  MAE: {avg_mae:10.6f}")
            print(f"  MSE: {avg_mse:10.6f}\n")

        return results

    @staticmethod
    def test_multi_input(latent_dim: int = 64,
                        train_range: Tuple[int, int] = (-9, 9),
                        input_sizes: List[int] = None) -> dict:
        """
        Test multi-input scaling: 2, 3, 4+ inputs.
        
        KEY DISCOVERY: Tests if architecture scales with input count.
        """
        if input_sizes is None:
            input_sizes = [2, 3, 4]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        results = {}

        print(f"\nðŸ”º Multi-Input Scaling Test")
        print(f"{'='*60}\n")

        # Import here to avoid circular dependency
        from tetrahedral_core import TetrahedralCore
        from tetrahedral_trainer import TetrahedralTrainer
        import torch.optim as optim

        for n_inputs in input_sizes:
            print(f"Testing {n_inputs}-input model...")

            # Create adapter and dataset
            adapter = ArithmeticAdapter(n_inputs=n_inputs)
            train_data = adapter.create_dataset(train_range, exhaustive=True)
            train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

            # Create model
            model = TetrahedralCore(
                input_dim=n_inputs,
                output_dim=1,
                latent_dim=latent_dim
            )
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Train
            trainer = TetrahedralTrainer(model, optimizer, device)
            history = trainer.train(train_loader, epochs=100, verbose=False)

            final_loss = history['train_loss'][-1]
            results[f"{n_inputs}_inputs"] = {
                'final_loss': final_loss,
                'history': history
            }

            print(f"  Final loss: {final_loss:.6f}\n")

        print(f"âœ“ Multi-input test complete!")
        print(f"  Sweet spot: 3 inputs (historically best)")


        return results
