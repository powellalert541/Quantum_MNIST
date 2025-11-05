"""
Optimized Configuration for Quantum MNIST Classification

This configuration includes several optimizations based on the baseline training results:
1. Increased quantum circuit depth for better expressivity
2. Adjusted learning rate schedule for better convergence
3. Increased training data for better generalization
4. Modified batch size for more stable gradients
5. Added gradient clipping to prevent exploding gradients
"""

import torch


class ConfigOptimized:
    """
    Optimized configuration class with improved hyperparameters.
    """

    # Random seed for reproducibility
    SEED = 42

    # Device configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data configuration
    DATA_DIR = './data'
    BATCH_SIZE = 24  # FASTER: Increased from 16 to 24 for faster training
    NUM_WORKERS = 0

    # Dataset type
    DATASET_TYPE = 'binary'

    # Binary classification settings
    BINARY_CLASS_A = 0
    BINARY_CLASS_B = 1
    BINARY_TRAIN_SIZE = 1500  # FASTER: Reduced from 2500 to 1500 samples per class
    BINARY_TEST_SIZE = 100

    # Multi-class settings
    MULTICLASS_CLASSES = [0, 1, 2]
    MULTICLASS_SAMPLES_PER_CLASS = 200

    # Model configuration
    MODEL_TYPE = 'hybrid'

    # OPTIMIZATION 3: Enhanced quantum circuit parameters (balanced for speed)
    N_QUBITS = 5  # FASTER: Reduced from 6 to 5 qubits (still better than baseline's 4)
    FEATURE_REPS = 2  # FASTER: Kept at 2 (same as baseline)
    VAR_REPS = 4      # Increased from 3 to 4 for more variational capacity

    # OPTIMIZATION 4: Improved training hyperparameters
    LEARNING_RATE = 0.005  # Reduced from 0.01 for more stable training
    NUM_EPOCHS = 25  # FASTER: Reduced from 30 to 25 epochs
    WEIGHT_DECAY = 5e-5  # Reduced from 1e-4 to reduce regularization

    # Optimizer settings
    OPTIMIZER = 'adam'
    MOMENTUM = 0.9

    # OPTIMIZATION 5: Better learning rate scheduler
    USE_SCHEDULER = True
    SCHEDULER_STEP_SIZE = 7  # Changed from 5 to 7 epochs
    SCHEDULER_GAMMA = 0.6    # Changed from 0.5 to 0.6 for gentler decay

    # OPTIMIZATION 6: Increased patience for early stopping
    EARLY_STOPPING = True
    PATIENCE = 8  # Increased from 5 to 8 to allow more exploration

    # OPTIMIZATION 7: Gradient clipping (new parameter)
    USE_GRADIENT_CLIPPING = True
    GRADIENT_CLIP_VALUE = 1.0

    # Checkpointing
    SAVE_CHECKPOINTS = True
    CHECKPOINT_DIR = './models_optimized'
    SAVE_BEST_ONLY = True

    # Logging
    LOG_DIR = './logs_optimized'
    LOG_INTERVAL = 10
    VERBOSE = True

    # Results
    RESULTS_DIR = './results_optimized'
    PLOTS_DIR = './results_optimized/plots'
    METRICS_DIR = './results_optimized/metrics'

    # Visualization
    SAVE_PLOTS = True
    PLOT_DPI = 300
    PLOT_FORMAT = 'png'

    @classmethod
    def get_num_classes(cls):
        """Get the number of classes based on dataset type."""
        if cls.DATASET_TYPE == 'binary':
            return 2
        else:
            return len(cls.MULTICLASS_CLASSES)

    @classmethod
    def print_config(cls):
        """Print all configuration settings in a readable format."""
        print("\n" + "=" * 60)
        print("OPTIMIZED CONFIGURATION SETTINGS")
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

        if cls.USE_GRADIENT_CLIPPING:
            print(f"  Gradient Clipping: Enabled (max_norm={cls.GRADIENT_CLIP_VALUE})")

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
        """Convert configuration to dictionary."""
        config_dict = {}
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                config_dict[attr] = getattr(cls, attr)
        return config_dict


# Create a default config instance
config_optimized = ConfigOptimized()


if __name__ == "__main__":
    ConfigOptimized.print_config()
    print(f"Number of classes: {ConfigOptimized.get_num_classes()}")

