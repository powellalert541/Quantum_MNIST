#!/usr/bin/env python3
"""
Compare Baseline and Optimized Model Results

This script compares the training results between the baseline and optimized models.
It generates comparison plots and prints a summary table.

Usage:
    python compare_results.py
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_history(filepath):
    """Load training history from JSON file."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_comparison(baseline_history, optimized_history, save_dir='./results_comparison'):
    """
    Create comparison plots between baseline and optimized models.
    
    Args:
        baseline_history: Training history from baseline model
        optimized_history: Training history from optimized model
        save_dir: Directory to save comparison plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Baseline vs Optimized Model Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    if baseline_history:
        epochs_baseline = range(1, len(baseline_history['train_loss']) + 1)
        ax.plot(epochs_baseline, baseline_history['train_loss'], 
                'b-', label='Baseline', linewidth=2)
    if optimized_history:
        epochs_optimized = range(1, len(optimized_history['train_loss']) + 1)
        ax.plot(epochs_optimized, optimized_history['train_loss'], 
                'r-', label='Optimized', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax = axes[0, 1]
    if baseline_history:
        ax.plot(epochs_baseline, baseline_history['val_loss'], 
                'b-', label='Baseline', linewidth=2)
    if optimized_history:
        ax.plot(epochs_optimized, optimized_history['val_loss'], 
                'r-', label='Optimized', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Training Accuracy
    ax = axes[1, 0]
    if baseline_history:
        ax.plot(epochs_baseline, baseline_history['train_acc'], 
                'b-', label='Baseline', linewidth=2)
    if optimized_history:
        ax.plot(epochs_optimized, optimized_history['train_acc'], 
                'r-', label='Optimized', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Validation Accuracy
    ax = axes[1, 1]
    if baseline_history:
        ax.plot(epochs_baseline, baseline_history['val_acc'], 
                'b-', label='Baseline', linewidth=2)
    if optimized_history:
        ax.plot(epochs_optimized, optimized_history['val_acc'], 
                'r-', label='Optimized', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'baseline_vs_optimized_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    plt.close()


def print_summary_table(baseline_history, optimized_history, 
                       baseline_results, optimized_results):
    """
    Print a summary comparison table.
    
    Args:
        baseline_history: Training history from baseline model
        optimized_history: Training history from optimized model
        baseline_results: Test results from baseline model
        optimized_results: Test results from optimized model
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    # Create comparison data
    data = []
    
    if baseline_history:
        data.append({
            'Model': 'Baseline',
            'Epochs': len(baseline_history['train_loss']),
            'Best Val Acc (%)': f"{max(baseline_history['val_acc']):.2f}",
            'Final Train Acc (%)': f"{baseline_history['train_acc'][-1]:.2f}",
            'Final Val Acc (%)': f"{baseline_history['val_acc'][-1]:.2f}",
            'Final Train Loss': f"{baseline_history['train_loss'][-1]:.4f}",
            'Final Val Loss': f"{baseline_history['val_loss'][-1]:.4f}",
            'Training Time (min)': f"{baseline_history.get('total_time', 0)/60:.1f}"
        })
    
    if optimized_history:
        data.append({
            'Model': 'Optimized',
            'Epochs': len(optimized_history['train_loss']),
            'Best Val Acc (%)': f"{max(optimized_history['val_acc']):.2f}",
            'Final Train Acc (%)': f"{optimized_history['train_acc'][-1]:.2f}",
            'Final Val Acc (%)': f"{optimized_history['val_acc'][-1]:.2f}",
            'Final Train Loss': f"{optimized_history['train_loss'][-1]:.4f}",
            'Final Val Loss': f"{optimized_history['val_loss'][-1]:.4f}",
            'Training Time (min)': f"{optimized_history.get('total_time', 0)/60:.1f}"
        })
    
    # Create DataFrame and print
    df = pd.DataFrame(data)
    print("\nTraining Metrics:")
    print(df.to_string(index=False))
    
    # Test results comparison
    if baseline_results or optimized_results:
        print("\n" + "-" * 80)
        print("Test Set Performance:")
        print("-" * 80)
        
        test_data = []
        if baseline_results:
            test_data.append({
                'Model': 'Baseline',
                'Test Accuracy (%)': f"{baseline_results.get('Accuracy', 0)*100:.2f}",
                'Precision': f"{baseline_results.get('Precision', 0):.4f}",
                'Recall': f"{baseline_results.get('Recall', 0):.4f}",
                'F1-Score': f"{baseline_results.get('F1-Score', 0):.4f}"
            })
        
        if optimized_results:
            test_data.append({
                'Model': 'Optimized',
                'Test Accuracy (%)': f"{optimized_results.get('Accuracy', 0)*100:.2f}",
                'Precision': f"{optimized_results.get('Precision', 0):.4f}",
                'Recall': f"{optimized_results.get('Recall', 0):.4f}",
                'F1-Score': f"{optimized_results.get('F1-Score', 0):.4f}"
            })
        
        test_df = pd.DataFrame(test_data)
        print(test_df.to_string(index=False))
    
    # Calculate improvements
    if baseline_history and optimized_history:
        print("\n" + "-" * 80)
        print("Improvements:")
        print("-" * 80)
        
        baseline_best_val = max(baseline_history['val_acc'])
        optimized_best_val = max(optimized_history['val_acc'])
        val_acc_improvement = optimized_best_val - baseline_best_val
        
        print(f"Validation Accuracy Improvement: {val_acc_improvement:+.2f}%")
        
        if baseline_results and optimized_results:
            test_acc_improvement = (optimized_results.get('Accuracy', 0) - 
                                   baseline_results.get('Accuracy', 0)) * 100
            f1_improvement = (optimized_results.get('F1-Score', 0) - 
                            baseline_results.get('F1-Score', 0))
            
            print(f"Test Accuracy Improvement: {test_acc_improvement:+.2f}%")
            print(f"F1-Score Improvement: {f1_improvement:+.4f}")
    
    print("=" * 80 + "\n")


def load_results_csv(filepath):
    """Load results from CSV file."""
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    if len(df) > 0:
        return df.iloc[0].to_dict()
    return None


def main():
    """Main comparison function."""
    print("\n" + "=" * 80)
    print("QUANTUM MNIST MODEL COMPARISON")
    print("=" * 80)
    
    # Load baseline results
    baseline_history = load_history('./logs/quantum_hybrid_history.json')
    baseline_results = load_results_csv('./results/metrics/results.csv')
    
    # Load optimized results
    optimized_history = load_history('./logs_optimized/quantum_hybrid_optimized_history.json')
    optimized_results = load_results_csv('./results_optimized/metrics/results_optimized.csv')
    
    # Check if we have data to compare
    if not baseline_history and not optimized_history:
        print("\nNo training history found!")
        print("Please run training first:")
        print("  - Baseline: python train_model.py")
        print("  - Optimized: python train_model_optimized.py")
        return
    
    if not baseline_history:
        print("\nWarning: Baseline results not found. Run: python train_model.py")
    
    if not optimized_history:
        print("\nWarning: Optimized results not found. Run: python train_model_optimized.py")
    
    # Generate comparison plots
    if baseline_history or optimized_history:
        print("\nGenerating comparison plots...")
        plot_comparison(baseline_history, optimized_history)
    
    # Print summary table
    print_summary_table(baseline_history, optimized_history, 
                       baseline_results, optimized_results)
    
    print("Comparison complete!")


if __name__ == "__main__":
    main()

