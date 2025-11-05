"""
Visualization and Evaluation Utilities

This module provides functions for evaluating model performance and creating
visualizations including training curves, confusion matrices, and ROC curves.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
import pandas as pd
import os


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation loss and accuracy curves.

    Training curves help us understand how the model learned over time.
    We look for smooth improvement and check if the model is overfitting
    (when validation performance is much worse than training performance).

    Args:
        history: Dictionary containing training history with keys:
                 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Plot confusion matrix as a heatmap.

    The confusion matrix shows which classes the model confuses with each other.
    Diagonal elements are correct predictions, off-diagonal are misclassifications.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names for labeling
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    # Normalize to show percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        square=True,
        cbar_kws={'label': 'Percentage (%)'},
        ax=ax
    )

    # Add count annotations in addition to percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(
                j + 0.5, i + 0.7,
                f'({cm[i, j]})',
                ha="center", va="center",
                color="gray", fontsize=9
            )

    if class_names:
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    return fig


def plot_roc_curve(y_true, y_scores, n_classes=2, save_path=None):
    """
    Plot ROC (Receiver Operating Characteristic) curve.

    ROC curves show the trade-off between true positive rate and false positive
    rate at different classification thresholds. The area under the curve (AUC)
    summarizes performance: higher is better (max = 1.0).

    Args:
        y_true: True labels
        y_scores: Predicted probabilities or scores
        n_classes: Number of classes
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if n_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr, tpr,
            color='darkorange',
            lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})'
        )
    else:
        # Multi-class classification
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr,
                lw=2,
                label=f'Class {i} (AUC = {roc_auc:.3f})'
            )

    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")

    return fig


def evaluate_model(model, data_loader, device, n_classes=2):
    """
    Evaluate model performance on a dataset.

    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader containing evaluation data
        device: Device to run evaluation on
        n_classes: Number of classes

    Returns:
        dict: Dictionary containing metrics and predictions
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get probabilities and predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            # Store results
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)

    # For multi-class, use weighted average
    avg_method = 'binary' if n_classes == 2 else 'weighted'
    precision = precision_score(all_labels, all_preds, average=avg_method, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=avg_method, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=avg_method, zero_division=0)

    # Get classification report
    report = classification_report(all_labels, all_preds, zero_division=0)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'classification_report': report
    }

    return results


def print_evaluation_results(results, model_name="Model"):
    """
    Print evaluation results in a formatted way.

    Args:
        results: Results dictionary from evaluate_model
        model_name: Name of the model for display
    """
    print("\n" + "=" * 60)
    print(f"{model_name} Evaluation Results")
    print("=" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    print("\n" + "-" * 60)
    print("Detailed Classification Report:")
    print("-" * 60)
    print(results['classification_report'])
    print("=" * 60 + "\n")


def compare_models(results_dict, save_path=None):
    """
    Create a comparison plot for multiple models.

    Args:
        results_dict: Dictionary mapping model names to their results
        save_path: Path to save the plot
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    model_names = list(results_dict.keys())

    # Collect data
    data = {metric: [] for metric in metrics}
    for model_name in model_names:
        for metric in metrics:
            data[metric].append(results_dict[model_name][metric])

    # Create plot
    x = np.arange(len(metrics))
    width = 0.35 if len(model_names) == 2 else 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model_name in enumerate(model_names):
        values = [data[metric][i] for metric in metrics]
        offset = width * (i - len(model_names)/2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=model_name)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9
            )

    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")

    return fig


def save_results_to_csv(results_dict, save_path):
    """
    Save evaluation results to a CSV file.

    Args:
        results_dict: Dictionary mapping model names to their results
        save_path: Path to save the CSV file
    """
    data = []
    for model_name, results in results_dict.items():
        data.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1_score']
        })

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    # Demonstration with dummy data
    print("Visualization Utilities Demonstration")
    print("=" * 60)

    # Create dummy training history
    epochs = 10
    history = {
        'train_loss': np.linspace(1.5, 0.3, epochs) + np.random.normal(0, 0.05, epochs),
        'val_loss': np.linspace(1.5, 0.4, epochs) + np.random.normal(0, 0.08, epochs),
        'train_acc': np.linspace(50, 95, epochs) + np.random.normal(0, 2, epochs),
        'val_acc': np.linspace(50, 92, epochs) + np.random.normal(0, 3, epochs),
    }

    # Plot training curves
    os.makedirs('../results/plots', exist_ok=True)
    plot_training_curves(history, save_path='../results/plots/demo_training_curves.png')

    # Create dummy predictions for confusion matrix
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = y_true.copy()
    # Add some errors
    errors = np.random.choice(n_samples, size=10, replace=False)
    y_pred[errors] = 1 - y_pred[errors]

    plot_confusion_matrix(
        y_true, y_pred,
        class_names=['Class 0', 'Class 1'],
        save_path='../results/plots/demo_confusion_matrix.png'
    )

    print("\nDemonstration complete. Check ../results/plots/ for output.")
