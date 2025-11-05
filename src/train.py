"""
Training Script with Logging and Checkpointing

This module handles the training loop, validation, and model checkpointing.
It provides reusable training functions that work with any PyTorch model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import json
import time
from datetime import datetime
from tqdm import tqdm


class Trainer:
    """
    Handles model training with logging, checkpointing, and early stopping.

    This class encapsulates all training logic, making it easy to train
    different models with consistent behavior.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        config,
        model_name='model'
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            config: Configuration object with training settings
            model_name: Name for saving checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.model_name = model_name

        # Setup scheduler if enabled
        self.scheduler = None
        if config.USE_SCHEDULER:
            self.scheduler = StepLR(
                optimizer,
                step_size=config.SCHEDULER_STEP_SIZE,
                gamma=config.SCHEDULER_GAMMA
            )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
        }

        # Early stopping
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0

        # Checkpointing
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)

    def train_epoch(self):
        """
        Train for one epoch.

        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Use tqdm for progress bar
        pbar = tqdm(self.train_loader, desc='Training', leave=False)

        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()

            # Gradient clipping if enabled
            if hasattr(self.config, 'USE_GRADIENT_CLIPPING') and self.config.USE_GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.GRADIENT_CLIP_VALUE
                )

            self.optimizer.step()

            # Calculate statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self):
        """
        Validate the model.

        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validation', leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Calculate statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest checkpoint
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f'{self.model_name}_checkpoint.pth'
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model if improved
        if is_best:
            best_path = os.path.join(
                self.config.CHECKPOINT_DIR,
                f'{self.model_name}_best.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"  New best model saved! Validation accuracy: {self.best_val_acc:.2f}%")

    def train(self, num_epochs):
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            dict: Training history
        """
        print(f"\nTraining {self.model_name}")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("=" * 60 + "\n")

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()

            # Train for one epoch
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Print progress
            print(f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s):")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")

            # Check if validation accuracy improved
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            # Save checkpoint
            if self.config.SAVE_CHECKPOINTS:
                if self.config.SAVE_BEST_ONLY:
                    if is_best:
                        self.save_checkpoint(epoch, is_best=True)
                else:
                    self.save_checkpoint(epoch, is_best)

            # Early stopping
            if self.config.EARLY_STOPPING:
                if self.epochs_no_improve >= self.config.PATIENCE:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    print(f"No improvement for {self.config.PATIENCE} consecutive epochs")
                    break

            print()

        # Training complete
        total_time = time.time() - start_time
        print("=" * 60)
        print("Training Complete!")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print("=" * 60 + "\n")

        # Save training history
        self.save_history()

        return self.history

    def save_history(self):
        """
        Save training history to JSON file.
        """
        history_path = os.path.join(
            self.config.LOG_DIR,
            f'{self.model_name}_history.json'
        )

        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

        print(f"Training history saved to {history_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.history = checkpoint.get('history', self.history)
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {checkpoint['epoch']}")


def create_optimizer(model, config):
    """
    Create optimizer based on configuration.

    Args:
        model: PyTorch model
        config: Configuration object

    Returns:
        Optimizer
    """
    if config.OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")

    return optimizer


if __name__ == "__main__":
    print("Training utilities module")
    print("This module provides the Trainer class for model training.")
    print("Import this module in your main training script.")
