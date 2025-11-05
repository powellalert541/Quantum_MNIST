#!/usr/bin/env python3
"""
Monitor Training Progress

This script monitors the training progress by reading the log file
and displaying key metrics in real-time.

Usage:
    python monitor_training.py
"""

import json
import os
import time
from datetime import datetime, timedelta


def load_history(filepath):
    """Load training history from JSON file."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None


def format_time(seconds):
    """Format seconds into human-readable time."""
    return str(timedelta(seconds=int(seconds)))


def print_progress(history, model_name="Model"):
    """Print training progress."""
    if not history:
        print(f"No training history found for {model_name}")
        return
    
    epochs_completed = len(history.get('train_loss', []))
    if epochs_completed == 0:
        print(f"{model_name}: Training not started yet")
        return
    
    print(f"\n{'='*80}")
    print(f"{model_name} Training Progress")
    print(f"{'='*80}")
    
    # Current epoch info
    print(f"\nEpochs Completed: {epochs_completed}")
    print(f"Latest Metrics:")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Train Acc:  {history['train_acc'][-1]:.2f}%")
    print(f"  Val Loss:   {history['val_loss'][-1]:.4f}")
    print(f"  Val Acc:    {history['val_acc'][-1]:.2f}%")
    
    if 'lr' in history and len(history['lr']) > 0:
        print(f"  Learning Rate: {history['lr'][-1]:.6f}")
    
    # Best metrics
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    # Time info
    if 'total_time' in history:
        total_time = history['total_time']
        avg_time_per_epoch = total_time / epochs_completed
        print(f"\nTime Elapsed: {format_time(total_time)}")
        print(f"Avg Time per Epoch: {format_time(avg_time_per_epoch)}")
    
    # Progress visualization
    print(f"\nTraining Progress:")
    print(f"Epoch | Train Loss | Train Acc | Val Loss | Val Acc")
    print(f"{'-'*60}")
    
    # Show last 5 epochs
    start_idx = max(0, epochs_completed - 5)
    for i in range(start_idx, epochs_completed):
        epoch_num = i + 1
        marker = " ✓" if history['val_acc'][i] == best_val_acc else ""
        print(f"{epoch_num:5d} | {history['train_loss'][i]:10.4f} | "
              f"{history['train_acc'][i]:9.2f}% | {history['val_loss'][i]:8.4f} | "
              f"{history['val_acc'][i]:7.2f}%{marker}")


def monitor_both_models():
    """Monitor both baseline and optimized models."""
    print("\n" + "="*80)
    print("QUANTUM MNIST TRAINING MONITOR")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check baseline model
    baseline_history = load_history('./logs/quantum_hybrid_history.json')
    if baseline_history:
        print_progress(baseline_history, "Baseline Model")
    else:
        print("\nBaseline Model: No training history found")
    
    # Check optimized model
    optimized_history = load_history('./logs_optimized/quantum_hybrid_optimized_history.json')
    if optimized_history:
        print_progress(optimized_history, "Optimized Model")
    else:
        print("\nOptimized Model: Training in progress or not started")
        print("(History file will be created after first epoch completes)")
    
    print("\n" + "="*80)


def continuous_monitor(interval=60):
    """
    Continuously monitor training progress.
    
    Args:
        interval: Update interval in seconds
    """
    print("Starting continuous monitoring...")
    print(f"Updating every {interval} seconds. Press Ctrl+C to stop.")
    
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            monitor_both_models()
            print(f"\nNext update in {interval} seconds...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        continuous_monitor(interval)
    else:
        monitor_both_models()
        print("\nTip: Run with --continuous flag for live monitoring:")
        print("  python monitor_training.py --continuous [interval_seconds]")


if __name__ == "__main__":
    main()

