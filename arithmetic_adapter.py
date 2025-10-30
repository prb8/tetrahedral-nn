"""
Arithmetic Adapter - The Gold Standard

The perfect test case for tetrahedral architecture.
Pure signal, no preprocessing, lossless data representation.

Key feature: Trains on [-9, 9], generalizes to trillions (float precision limited).
"""

import torch
from torch.utils.data import TensorDataset
from typing import Tuple
import itertools


class ArithmeticAdapter:
    """
    THE GOLD STANDARD ADAPTER
    
    This is as simple as it gets:
        - Input: n numbers
        - Output: 1 number (their sum)
        - NO preprocessing
        - NO feature engineering
        - Just pure, lossless signal
    
    This adapter is PERFECT - use it as the template for all others!
    """
    def __init__(self, n_inputs: int = 2):
        self.n_inputs = n_inputs
        self.input_dim = n_inputs
        self.output_dim = 1

    def create_dataset(self,
                      train_range: Tuple[int, int] = (-9, 9),
                      exhaustive: bool = True) -> TensorDataset:
        """
        Create arithmetic dataset.
        
        Args:
            train_range: Range of numbers to use (inclusive)
            exhaustive: If True, use all combinations. If False, sample randomly.
        
        Returns:
            TensorDataset ready for training
        """
        if exhaustive:
            # Exhaustive dataset - all combinations
            inputs = []
            targets = []

            ranges = [range(train_range[0], train_range[1] + 1)] * self.n_inputs

            for combo in itertools.product(*ranges):
                inputs.append(list(combo))
                targets.append([sum(combo)])  # Sum by default

            inputs = torch.tensor(inputs, dtype=torch.float32)
            targets = torch.tensor(targets, dtype=torch.float32)

            print(f"âœ“ Exhaustive dataset: {len(inputs):,} samples")
            print(f"  Range: [{train_range[0]}, {train_range[1]}]")
            print(f"  Inputs: {self.n_inputs}")

        else:
            # Random sampling
            n_samples = 10000
            inputs = torch.randint(train_range[0], train_range[1] + 1,
                                 (n_samples, self.n_inputs)).float()
            targets = inputs.sum(dim=1, keepdim=True)

            print(f"âœ“ Random dataset: {len(inputs):,} samples")

        return TensorDataset(inputs, targets)

    def create_test_dataset(self,
                          test_range: Tuple[int, int],
                          n_samples: int = 1000) -> TensorDataset:
        """
        Create test dataset for generalization testing.
        
        This is THE key test - can the model generalize beyond training range?
        """
        inputs = torch.randint(test_range[0], test_range[1] + 1,
                             (n_samples, self.n_inputs)).float()
        targets = inputs.sum(dim=1, keepdim=True)

        print(f"âœ“ Test dataset: {n_samples:,} samples")
        print(f"  Range: [{test_range[0]}, {test_range[1]}]")

        return TensorDataset(inputs, targets)

    def verify_output(self, input_vals: list, output: float) -> bool:
        """
        Verify that output equals sum of inputs.
        Useful for checking if model learned correctly.
        """
        expected = sum(input_vals)
        error = abs(output - expected)
        return error < 0.001  # Allow small floating point error
