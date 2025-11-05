"""
Data Loading and Preprocessing Utilities for MNIST

This module handles loading the MNIST dataset, preprocessing images,
and creating train/validation/test splits suitable for quantum machine learning.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import os


class MNISTDataLoader:
    """
    Handles loading and preprocessing of MNIST data for quantum ML.

    For practical training on simulators, we often need to use subsets of the
    full dataset. This class provides flexible options for creating smaller
    datasets while maintaining class balance.
    """

    def __init__(self, data_dir='./data', batch_size=32, num_workers=0):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory to store/load MNIST data
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Standard normalization for MNIST
        # These values center the data around 0 with standard deviation 1
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def load_full_dataset(self):
        """
        Load the complete MNIST dataset.

        Returns:
            tuple: (train_dataset, test_dataset)
        """
        train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )

        test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )

        return train_dataset, test_dataset

    def create_binary_classification_dataset(self, class_a=0, class_b=1,
                                             train_size=500, test_size=100):
        """
        Create a binary classification dataset with two specific digits.

        Binary classification is faster to train and good for initial experiments.
        We filter the dataset to only include two classes.

        Args:
            class_a: First class label
            class_b: Second class label
            train_size: Number of training samples per class
            test_size: Number of test samples per class

        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        train_dataset, test_dataset = self.load_full_dataset()

        # Filter training data for the two classes
        train_indices = []
        count_a, count_b = 0, 0

        for idx, (_, label) in enumerate(train_dataset):
            if label == class_a and count_a < train_size:
                train_indices.append(idx)
                count_a += 1
            elif label == class_b and count_b < train_size:
                train_indices.append(idx)
                count_b += 1

            if count_a >= train_size and count_b >= train_size:
                break

        # Filter test data
        test_indices = []
        count_a, count_b = 0, 0

        for idx, (_, label) in enumerate(test_dataset):
            if label == class_a and count_a < test_size:
                test_indices.append(idx)
                count_a += 1
            elif label == class_b and count_b < test_size:
                test_indices.append(idx)
                count_b += 1

            if count_a >= test_size and count_b >= test_size:
                break

        # Create subsets
        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(test_dataset, test_indices)

        # Split training into train and validation (80/20 split)
        train_size_split = int(0.8 * len(train_subset))
        val_size_split = len(train_subset) - train_size_split
        train_subset, val_subset = random_split(
            train_subset,
            [train_size_split, val_size_split],
            generator=torch.Generator().manual_seed(42)
        )

        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        return train_loader, val_loader, test_loader

    def create_multiclass_dataset(self, classes=None, samples_per_class=200):
        """
        Create a multi-class classification dataset.

        Args:
            classes: List of class labels to include (None for all 10 classes)
            samples_per_class: Number of training samples per class

        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        if classes is None:
            classes = list(range(10))

        train_dataset, test_dataset = self.load_full_dataset()

        # Filter training data
        train_indices = []
        class_counts = {c: 0 for c in classes}

        for idx, (_, label) in enumerate(train_dataset):
            if label in classes and class_counts[label] < samples_per_class:
                train_indices.append(idx)
                class_counts[label] += 1

            if all(count >= samples_per_class for count in class_counts.values()):
                break

        # Filter test data (use fewer samples for testing)
        test_samples_per_class = samples_per_class // 5
        test_indices = []
        class_counts = {c: 0 for c in classes}

        for idx, (_, label) in enumerate(test_dataset):
            if label in classes and class_counts[label] < test_samples_per_class:
                test_indices.append(idx)
                class_counts[label] += 1

            if all(count >= test_samples_per_class for count in class_counts.values()):
                break

        # Create subsets
        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(test_dataset, test_indices)

        # Split training into train and validation
        train_size_split = int(0.8 * len(train_subset))
        val_size_split = len(train_subset) - train_size_split
        train_subset, val_subset = random_split(
            train_subset,
            [train_size_split, val_size_split],
            generator=torch.Generator().manual_seed(42)
        )

        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        return train_loader, val_loader, test_loader

    def get_dataset_info(self, data_loader):
        """
        Print information about a dataset.

        Args:
            data_loader: PyTorch DataLoader to analyze
        """
        # Count samples per class
        class_counts = {}
        total_samples = 0

        for _, labels in data_loader:
            for label in labels:
                label_item = label.item()
                class_counts[label_item] = class_counts.get(label_item, 0) + 1
                total_samples += 1

        print(f"\nDataset Information:")
        print(f"Total samples: {total_samples}")
        print(f"Number of classes: {len(class_counts)}")
        print(f"Batch size: {data_loader.batch_size}")
        print(f"Number of batches: {len(data_loader)}")
        print(f"\nClass distribution:")
        for class_label in sorted(class_counts.keys()):
            print(f"  Class {class_label}: {class_counts[class_label]} samples")


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    Setting seeds ensures that we get consistent results across runs,
    which is important for comparing different models and hyperparameters.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get the best available device (CPU or GPU).

    For quantum simulations, CPU is typically sufficient and often preferred
    since quantum simulators are CPU-optimized.

    Returns:
        torch.device: The device to use for training
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


if __name__ == "__main__":
    # Demonstration of data loading
    print("MNIST Data Loading Demonstration")
    print("=" * 60)

    # Set seed for reproducibility
    set_seed(42)

    # Create data loader
    data_loader = MNISTDataLoader(batch_size=32)

    # Create binary classification dataset (0 vs 1)
    print("\n1. Creating Binary Classification Dataset (0 vs 1)")
    print("-" * 60)
    train_loader, val_loader, test_loader = data_loader.create_binary_classification_dataset(
        class_a=0,
        class_b=1,
        train_size=500,
        test_size=100
    )

    print("\nTraining Set:")
    data_loader.get_dataset_info(train_loader)

    print("\nValidation Set:")
    data_loader.get_dataset_info(val_loader)

    print("\nTest Set:")
    data_loader.get_dataset_info(test_loader)

    # Create multi-class dataset
    print("\n" + "=" * 60)
    print("2. Creating Multi-Class Dataset (0, 1, 2)")
    print("-" * 60)
    train_loader, val_loader, test_loader = data_loader.create_multiclass_dataset(
        classes=[0, 1, 2],
        samples_per_class=200
    )

    print("\nTraining Set:")
    data_loader.get_dataset_info(train_loader)

    # Get device information
    print("\n" + "=" * 60)
    print("Device Information:")
    print("-" * 60)
    device = get_device()
