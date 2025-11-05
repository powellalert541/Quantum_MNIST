#!/usr/bin/env python3
"""
Main Training Script for Quantum MNIST Classification

This script trains both quantum and classical models on MNIST data and
compares their performance. Run this script to train models end-to-end.

Usage:
    python train_model.py
"""

import sys
import os
import torch
import torch.nn as nn

# Add src directory to path
sys.path.append('src')

from config import Config
from data_utils import MNISTDataLoader, set_seed, get_device
from models import SimplifiedHybridQNN, ClassicalNN, model_summary
from train import Trainer, create_optimizer
from visualize import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    evaluate_model,
    print_evaluation_results,
    compare_models,
    save_results_to_csv
)


def train_quantum_model(config, train_loader, val_loader, device):
    """
    Train the quantum hybrid model.

    Args:
        config: Configuration object
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on

    Returns:
        tuple: (model, history)
    """
    print("\n" + "=" * 60)
    print("TRAINING QUANTUM HYBRID MODEL")
    print("=" * 60)

    # Create model
    n_classes = config.get_num_classes()
    model = SimplifiedHybridQNN(
        n_qubits=config.N_QUBITS,
        n_classes=n_classes
    ).to(device)

    # Print model summary
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
        model_name='quantum_hybrid'
    )

    # Train the model
    history = trainer.train(config.NUM_EPOCHS)

    return model, history


def train_classical_model(config, train_loader, val_loader, device):
    """
    Train the classical baseline model.

    Args:
        config: Configuration object
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on

    Returns:
        tuple: (model, history)
    """
    print("\n" + "=" * 60)
    print("TRAINING CLASSICAL BASELINE MODEL")
    print("=" * 60)

    # Create model
    n_classes = config.get_num_classes()
    model = ClassicalNN(
        hidden_sizes=[128, 64, 32],
        n_classes=n_classes
    ).to(device)

    # Print model summary
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
        model_name='classical_baseline'
    )

    # Train the model
    history = trainer.train(config.NUM_EPOCHS)

    return model, history


def main():
    """
    Main function to run the complete training pipeline.
    """
    # Print header
    print("\n" + "=" * 60)
    print("QUANTUM MNIST CLASSIFICATION")
    print("Hybrid Quantum-Classical Neural Network Training")
    print("=" * 60)

    # Print configuration
    Config.print_config()

    # Set random seed for reproducibility
    set_seed(Config.SEED)

    # Get device
    device = get_device()

    # Create data loaders
    print("\nLoading MNIST data...")
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

    data_loader.get_dataset_info(train_loader)

    # Dictionary to store results
    all_results = {}
    all_histories = {}

    # Train quantum model if specified
    if Config.MODEL_TYPE in ['hybrid', 'both']:
        try:
            quantum_model, quantum_history = train_quantum_model(
                Config, train_loader, val_loader, device
            )
            all_histories['Quantum Hybrid'] = quantum_history

            # Evaluate on test set
            print("\nEvaluating Quantum Model on Test Set...")
            quantum_results = evaluate_model(
                quantum_model, test_loader, device, Config.get_num_classes()
            )
            all_results['Quantum Hybrid'] = quantum_results
            print_evaluation_results(quantum_results, "Quantum Hybrid Model")

        except Exception as e:
            print(f"\nError training quantum model: {e}")
            print("This may be due to Qiskit setup. Continuing with classical model only...")

    # Train classical model
    if Config.MODEL_TYPE in ['classical', 'both'] or Config.MODEL_TYPE == 'hybrid' and len(all_results) == 0:
        classical_model, classical_history = train_classical_model(
            Config, train_loader, val_loader, device
        )
        all_histories['Classical'] = classical_history

        # Evaluate on test set
        print("\nEvaluating Classical Model on Test Set...")
        classical_results = evaluate_model(
            classical_model, test_loader, device, Config.get_num_classes()
        )
        all_results['Classical'] = classical_results
        print_evaluation_results(classical_results, "Classical Model")

    # Generate visualizations
    if Config.SAVE_PLOTS and len(all_results) > 0:
        print("\nGenerating visualizations...")
        os.makedirs(Config.PLOTS_DIR, exist_ok=True)
        os.makedirs(Config.METRICS_DIR, exist_ok=True)

        # Plot training curves for each model
        for model_name, history in all_histories.items():
            plot_path = os.path.join(
                Config.PLOTS_DIR,
                f'{model_name.lower().replace(" ", "_")}_training_curves.png'
            )
            plot_training_curves(history, save_path=plot_path)

        # Plot confusion matrices
        for model_name, results in all_results.items():
            cm_path = os.path.join(
                Config.PLOTS_DIR,
                f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
            )
            plot_confusion_matrix(
                results['labels'],
                results['predictions'],
                save_path=cm_path
            )

        # Plot ROC curves
        for model_name, results in all_results.items():
            roc_path = os.path.join(
                Config.PLOTS_DIR,
                f'{model_name.lower().replace(" ", "_")}_roc_curve.png'
            )
            plot_roc_curve(
                results['labels'],
                results['probabilities'],
                n_classes=Config.get_num_classes(),
                save_path=roc_path
            )

        # Compare models if we have multiple
        if len(all_results) > 1:
            comparison_path = os.path.join(Config.PLOTS_DIR, 'model_comparison.png')
            compare_models(all_results, save_path=comparison_path)

        # Save results to CSV
        csv_path = os.path.join(Config.METRICS_DIR, 'results.csv')
        save_results_to_csv(all_results, csv_path)

    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nResults Summary:")
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        print(f"  Test Accuracy: {results['accuracy']*100:.2f}%")
        print(f"  F1-Score: {results['f1_score']:.4f}")

    print("\n" + "=" * 60)
    print("Output files saved to:")
    print(f"  Models: {Config.CHECKPOINT_DIR}/")
    print(f"  Plots: {Config.PLOTS_DIR}/")
    print(f"  Metrics: {Config.METRICS_DIR}/")
    print(f"  Logs: {Config.LOG_DIR}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
