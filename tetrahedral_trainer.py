"""
Universal Training System for Tetrahedral Models

Works for all experiments: arithmetic, decomposition, images, video, etc.
The adapter handles data representation, the core handles processing,
this handles training dynamics.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Optional


class TetrahedralTrainer:
    """
    Universal training system for tetrahedral models.
    
    This is the same training loop for ALL experiments.
    """
    def __init__(self,
                 model,
                 optimizer: optim.Optimizer,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'test_mae': []
        }

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(inputs)
            loss = F.mse_loss(predictions, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0
        total_mae = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(inputs)
                loss = F.mse_loss(predictions, targets)
                mae = torch.abs(predictions - targets).mean()

                total_loss += loss.item()
                total_mae += mae.item()

        return total_loss / len(dataloader), total_mae / len(dataloader)

    def train(self,
             train_loader: DataLoader,
             test_loader: Optional[DataLoader] = None,
             epochs: int = 100,
             verbose: bool = True) -> dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data
            test_loader: Optional test data for evaluation
            epochs: Number of training epochs
            verbose: Print progress
        
        Returns:
            Training history dictionary
        """
        print(f"\nðŸ”º Training Tetrahedral Model")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Training samples: {len(train_loader.dataset):,}")
        if test_loader:
            print(f"Test samples: {len(test_loader.dataset):,}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            if test_loader and epoch % 10 == 0:
                test_loss, test_mae = self.evaluate(test_loader)
                self.history['test_loss'].append(test_loss)
                self.history['test_mae'].append(test_mae)

                if verbose:
                    print(f"Epoch {epoch+1:3d}/{epochs} | "
                          f"Train Loss: {train_loss:.6f} | "
                          f"Test Loss: {test_loss:.6f} | "
                          f"Test MAE: {test_mae:.6f}")
            elif verbose and epoch % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.6f}")

        print(f"\nâœ“ Training complete!")
        print(f"Final train loss: {self.history['train_loss'][-1]:.6f}")
        if test_loader:
            print(f"Final test MAE: {self.history['test_mae'][-1]:.6f}")

        return self.history

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        print(f"âœ“ Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']

        print(f"âœ“ Checkpoint loaded: {path}")
