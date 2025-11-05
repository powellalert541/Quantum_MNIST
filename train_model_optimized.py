#!/usr/bin/env python3
"""
Optimized Training Script for Quantum MNIST Classification

This script uses the optimized configuration (config_optimized.py) to train
the quantum hybrid model with improved hyperparameters.

Usage:
    python train_model_optimized.py
"""

import sys
import os
import torch
import torch.nn as nn

# Add src directory to path
sys.path.append('src')

from config_optimized import ConfigOptimized as Config
from data_utils import MNISTDataLoader, set_seed, get_device
from models import SimplifiedHybridQNN, model_summary
from train import Trainer, create_optimizer
from visualize import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    evaluate_model,
    print_evaluation_results,
    save_results_to_csv
)


def train_quantum_model(config, train_loader, val_loader, device):
    """
    Train the quantum hybrid model with optimized configuration.

    Args:
        config: Configuration object
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on

    Returns:
        tuple: (model, history)
    """
    print("\n" + "=" * 60)
    print("TRAINING OPTIMIZED QUANTUM HYBRID MODEL")
    print("=" * 60)

    # Initialize model
    model = SimplifiedHybridQNN(
        n_qubits=config.N_QUBITS,
        n_classes=config.get_num_classes(),
        feature_reps=config.FEATURE_REPS,
        var_reps=config.VAR_REPS
    ).to(device)

    # Print model summary
    print(f"\nModel: {model.__class__.__name__}")
    model_summary(model)

    # Create optimizer and loss function
    optimizer = create_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config,
        model_name='quantum_hybrid_optimized'
    )

    # Print gradient clipping status
    if hasattr(config, 'USE_GRADIENT_CLIPPING') and config.USE_GRADIENT_CLIPPING:
        print(f"\nGradient clipping enabled: max_norm={config.GRADIENT_CLIP_VALUE}")

    # Train model
    print(f"\nTraining quantum_hybrid_optimized")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Number of epochs: {config.NUM_EPOCHS}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print("=" * 60 + "\n")

    history = trainer.train(config.NUM_EPOCHS)

    return model, history


def main():
    """Main training function."""
    print("\n" + "=" * 60)
    print("QUANTUM MNIST CLASSIFICATION - OPTIMIZED VERSION")
    print("Hybrid Quantum-Classical Neural Network Training")
    print("=" * 60)

    # Print configuration
    Config.print_config()

    # Set random seed
    set_seed(Config.SEED)

    # Get device
    device = get_device()
    print(f"\nUsing {device}\n")

    # Create necessary directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.METRICS_DIR, exist_ok=True)

    # Load data
    print("Loading MNIST data...")
    data_loader = MNISTDataLoader(
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS
    )

    if Config.DATASET_TYPE == 'binary':
        train_loader, val_loader, test_loader = data_loader.create_binary_classification_dataset(
            class_a=Config.BINARY_CLASS_A,
            class_b=Config.BINARY_CLASS_B,
            train_size=Config.BINARY_TRAIN_SIZE,
            test_size=Config.BINARY_TEST_SIZE
        )
        print(f"\nBinary classification: {Config.BINARY_CLASS_A} vs {Config.BINARY_CLASS_B}")
    else:
        train_loader, val_loader, test_loader = data_loader.create_multiclass_dataset(
            classes=Config.MULTICLASS_CLASSES,
            samples_per_class=Config.MULTICLASS_SAMPLES_PER_CLASS
        )
        print(f"\nMulti-class classification: {Config.MULTICLASS_CLASSES}")

    # Print dataset info
    print(f"\nDataset Information:")
    print(f"Total samples: {len(train_loader.dataset)}")
    print(f"Number of classes: {Config.get_num_classes()}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Number of batches: {len(train_loader)}")

    # Get class distribution
    labels = [label for _, label in train_loader.dataset]
    unique, counts = torch.unique(torch.tensor(labels), return_counts=True)
    print(f"\nClass distribution:")
    for cls, count in zip(unique.tolist(), counts.tolist()):
        print(f"  Class {cls}: {count} samples")

    # Train quantum model
    quantum_model, quantum_history = train_quantum_model(
        Config, train_loader, val_loader, device
    )

    # Print training summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Total time: {quantum_history['total_time']:.1f}s ({quantum_history['total_time']/60:.1f} minutes)")
    print(f"Best validation accuracy: {max(quantum_history['val_acc']):.2f}%")
    print("=" * 60)

    # Save training history
    history_path = os.path.join(Config.LOG_DIR, 'quantum_hybrid_optimized_history.json')
    import json
    with open(history_path, 'w') as f:
        json.dump(quantum_history, f, indent=4)
    print(f"\nTraining history saved to {history_path}")

    # Evaluate on test set
    print("\nEvaluating Optimized Quantum Model on Test Set...")
    quantum_metrics = evaluate_model(quantum_model, test_loader, device)

    # Print results
    print("\n" + "=" * 60)
    print("Optimized Quantum Hybrid Model Evaluation Results")
    print("=" * 60)
    print_evaluation_results(quantum_metrics)

    # Generate visualizations
    if Config.SAVE_PLOTS:
        print("\nGenerating visualizations...")

        # Training curves
        plot_path = os.path.join(Config.PLOTS_DIR, 'quantum_hybrid_optimized_training_curves.png')
        plot_training_curves(quantum_history, save_path=plot_path)
        print(f"Training curves saved to {plot_path}")

        # Confusion matrix
        cm_path = os.path.join(Config.PLOTS_DIR, 'quantum_hybrid_optimized_confusion_matrix.png')
        plot_confusion_matrix(
            quantum_metrics['y_true'],
            quantum_metrics['y_pred'],
            classes=[str(i) for i in range(Config.get_num_classes())],
            save_path=cm_path
        )
        print(f"Confusion matrix saved to {cm_path}")

        # ROC curve (for binary classification)
        if Config.get_num_classes() == 2:
            roc_path = os.path.join(Config.PLOTS_DIR, 'quantum_hybrid_optimized_roc_curve.png')
            plot_roc_curve(
                quantum_metrics['y_true'],
                quantum_metrics['y_proba'][:, 1],
                save_path=roc_path
            )
            print(f"ROC curve saved to {roc_path}")

    # Save results to CSV
    results_csv = os.path.join(Config.METRICS_DIR, 'results_optimized.csv')
    save_results_to_csv(
        {'Quantum Hybrid Optimized': quantum_metrics},
        save_path=results_csv
    )
    print(f"Results saved to {results_csv}")

    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nResults Summary:")
    print(f"\nQuantum Hybrid Optimized:")
    print(f"  Test Accuracy: {quantum_metrics['accuracy']*100:.2f}%")
    print(f"  F1-Score: {quantum_metrics['f1']:.4f}")
    print("\n" + "=" * 60)
    print("Output files saved to:")
    print(f"  Models: {Config.CHECKPOINT_DIR}/")
    print(f"  Plots: {Config.PLOTS_DIR}/")
    print(f"  Metrics: {Config.METRICS_DIR}/")
    print(f"  Logs: {Config.LOG_DIR}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

