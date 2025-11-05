"""
Configuration file for Quantum MNIST Classification

This file contains all hyperparameters and settings for training.
Centralizing configuration makes it easy to experiment with different settings
without modifying the main training code.
"""

import torch


class Config:
    """
    Configuration class containing all hyperparameters and settings.
    """

    # Random seed for reproducibility
    SEED = 42

    # Device configuration
    # Set to 'cuda' to use GPU if available, or 'cpu' to force CPU usage
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data configuration
    DATA_DIR = './data'
    BATCH_SIZE = 32
    NUM_WORKERS = 0  # Number of workers for data loading (0 for main thread)

    # Dataset type: 'binary' or 'multiclass'
    DATASET_TYPE = 'binary'

    # Binary classification settings (only used if DATASET_TYPE='binary')
    BINARY_CLASS_A = 0
    BINARY_CLASS_B = 1
    BINARY_TRAIN_SIZE = 500  # Samples per class for training
    BINARY_TEST_SIZE = 100   # Samples per class for testing

    # Multi-class settings (only used if DATASET_TYPE='multiclass')
    MULTICLASS_CLASSES = [0, 1, 2]  # Which digits to classify
    MULTICLASS_SAMPLES_PER_CLASS = 200

    # Model configuration
    MODEL_TYPE = 'hybrid'  # 'hybrid' or 'classical'

    # Quantum circuit parameters (only used for hybrid model)
    N_QUBITS = 4
    FEATURE_REPS = 2  # Repetitions in feature map
    VAR_REPS = 3      # Repetitions in variational circuit

    # Training hyperparameters
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 20
    WEIGHT_DECAY = 1e-4  # L2 regularization

    # Optimizer settings
    OPTIMIZER = 'adam'  # 'adam' or 'sgd'
    MOMENTUM = 0.9      # Only used for SGD

    # Learning rate scheduler
    USE_SCHEDULER = True
    SCHEDULER_STEP_SIZE = 5  # Reduce LR every N epochs
    SCHEDULER_GAMMA = 0.5    # Multiply LR by this factor

    # Early stopping
    EARLY_STOPPING = True
    PATIENCE = 5  # Stop if no improvement for N epochs

    # Checkpointing
    SAVE_CHECKPOINTS = True
    CHECKPOINT_DIR = './models'
    SAVE_BEST_ONLY = True  # Only save model if validation accuracy improves

    # Logging
    LOG_DIR = './logs'
    LOG_INTERVAL = 10  # Log every N batches during training
    VERBOSE = True

    # Results
    RESULTS_DIR = './results'
    PLOTS_DIR = './results/plots'
    METRICS_DIR = './results/metrics'

    # Visualization
    SAVE_PLOTS = True
    PLOT_DPI = 300
    PLOT_FORMAT = 'png'

    @classmethod
    def get_num_classes(cls):
        """
        Get the number of classes based on dataset type.

        Returns:
            int: Number of classes
        """
        if cls.DATASET_TYPE == 'binary':
            return 2
        else:
            return len(cls.MULTICLASS_CLASSES)

    @classmethod
    def print_config(cls):
        """
        Print all configuration settings in a readable format.
        """
        print("\n" + "=" * 60)
        print("CONFIGURATION SETTINGS")
        print("=" * 60)

        print("\nGeneral Settings:")
        print(f"  Seed: {cls.SEED}")
        print(f"  Device: {cls.DEVICE}")

        print("\nData Settings:")
        print(f"  Data Directory: {cls.DATA_DIR}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Dataset Type: {cls.DATASET_TYPE}")

        if cls.DATASET_TYPE == 'binary':
            print(f"  Binary Classes: {cls.BINARY_CLASS_A} vs {cls.BINARY_CLASS_B}")
            print(f"  Training Samples per Class: {cls.BINARY_TRAIN_SIZE}")
            print(f"  Test Samples per Class: {cls.BINARY_TEST_SIZE}")
        else:
            print(f"  Classes: {cls.MULTICLASS_CLASSES}")
            print(f"  Samples per Class: {cls.MULTICLASS_SAMPLES_PER_CLASS}")

        print("\nModel Settings:")
        print(f"  Model Type: {cls.MODEL_TYPE}")
        if cls.MODEL_TYPE == 'hybrid':
            print(f"  Number of Qubits: {cls.N_QUBITS}")
            print(f"  Feature Map Reps: {cls.FEATURE_REPS}")
            print(f"  Variational Reps: {cls.VAR_REPS}")

        print("\nTraining Settings:")
        print(f"  Learning Rate: {cls.LEARNING_RATE}")
        print(f"  Number of Epochs: {cls.NUM_EPOCHS}")
        print(f"  Optimizer: {cls.OPTIMIZER}")
        print(f"  Weight Decay: {cls.WEIGHT_DECAY}")

        if cls.USE_SCHEDULER:
            print(f"  LR Scheduler: Enabled")
            print(f"    Step Size: {cls.SCHEDULER_STEP_SIZE}")
            print(f"    Gamma: {cls.SCHEDULER_GAMMA}")

        if cls.EARLY_STOPPING:
            print(f"  Early Stopping: Enabled (patience={cls.PATIENCE})")

        print("\nOutput Settings:")
        print(f"  Checkpoint Directory: {cls.CHECKPOINT_DIR}")
        print(f"  Results Directory: {cls.RESULTS_DIR}")
        print(f"  Save Plots: {cls.SAVE_PLOTS}")

        print("=" * 60 + "\n")

    @classmethod
    def to_dict(cls):
        """
        Convert configuration to dictionary.

        Returns:
            dict: Configuration as dictionary
        """
        config_dict = {}
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                config_dict[attr] = getattr(cls, attr)
        return config_dict


# Create a default config instance for easy import
config = Config()


if __name__ == "__main__":
    # Print configuration when run as a script
    Config.print_config()

    # Example: accessing configuration
    print(f"Number of classes for current dataset: {Config.get_num_classes()}")
