"""
Hybrid Quantum-Classical Neural Network Models

This module implements both quantum and classical models for MNIST classification.
The hybrid model combines quantum circuits with classical neural network layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from quantum_circuit import HybridQuantumCircuit


class QuantumLayer(nn.Module):
    """
    A quantum layer that can be integrated into a PyTorch neural network.

    This layer wraps a parameterized quantum circuit and allows it to be trained
    using standard PyTorch optimizers. The quantum circuit processes input features
    and produces output that can be fed to classical layers.
    """

    def __init__(self, n_qubits=4, quantum_params=None, feature_reps=2, var_reps=3):
        """
        Initialize the quantum layer.

        Args:
            n_qubits: Number of qubits in the quantum circuit
            quantum_params: Optional initial parameters for the quantum circuit
            feature_reps: Number of repetitions in the feature map
            var_reps: Number of repetitions in the variational circuit
        """
        super().__init__()
        self.n_qubits = n_qubits

        # Create the quantum circuit
        hybrid_qc = HybridQuantumCircuit(
            n_qubits=n_qubits,
            feature_dim=n_qubits,
            feature_reps=feature_reps,
            var_reps=var_reps
        )
        self.circuit = hybrid_qc.create_circuit()

        # Define how to interpret the measurement results
        # We use parity mapping: count the parity of measured bitstrings
        def parity(x):
            """Calculate parity of bitstring (0 for even number of 1s, 1 for odd)."""
            return np.sum(x) % 2

        # Create the quantum neural network
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.circuit.parameters[:n_qubits],
            weight_params=self.circuit.parameters[n_qubits:],
            interpret=parity,
            output_shape=2
        )

        # Wrap the QNN to make it compatible with PyTorch
        self.quantum_layer = TorchConnector(self.qnn)

    def forward(self, x):
        """
        Forward pass through the quantum layer.

        Args:
            x: Input tensor of shape (batch_size, n_qubits)

        Returns:
            Output tensor from quantum processing
        """
        return self.quantum_layer(x)


class HybridQNN(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network for MNIST classification.

    Architecture:
    1. Classical preprocessing layer: Reduces 28x28 image to 4 features
    2. Quantum layer: Processes features through parameterized quantum circuit
    3. Classical postprocessing layer: Maps quantum output to class predictions

    This architecture allows us to leverage both classical and quantum computing
    strengths. Classical layers handle high-dimensional data efficiently, while
    the quantum layer can potentially capture complex patterns.
    """

    def __init__(self, n_qubits=4, n_classes=10):
        """
        Initialize the hybrid quantum-classical network.

        Args:
            n_qubits: Number of qubits in quantum circuit
            n_classes: Number of output classes
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_classes = n_classes

        # Classical preprocessing: compress 784 dimensions to n_qubits
        # We use multiple layers to gradually reduce dimensionality
        self.pre_net = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, n_qubits),
            nn.Tanh()  # Output in [-1, 1] for quantum encoding
        )

        # Quantum processing layer
        self.quantum_layer = QuantumLayer(n_qubits=n_qubits)

        # Classical postprocessing: map quantum output to class predictions
        self.post_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        """
        Forward pass through the hybrid network.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) or (batch_size, 784)

        Returns:
            Class logits of shape (batch_size, n_classes)
        """
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Classical preprocessing
        x = self.pre_net(x)

        # Quantum processing
        x = self.quantum_layer(x)

        # Classical postprocessing
        x = self.post_net(x)

        return x


class ClassicalNN(nn.Module):
    """
    Classical neural network baseline for comparison.

    This is a standard fully connected neural network that serves as a baseline
    to compare against the quantum model. It has a similar number of parameters
    to make the comparison fair.
    """

    def __init__(self, input_size=784, hidden_sizes=[128, 64, 32], n_classes=10):
        """
        Initialize the classical neural network.

        Args:
            input_size: Size of input (784 for flattened MNIST)
            hidden_sizes: List of hidden layer sizes
            n_classes: Number of output classes
        """
        super().__init__()

        # Build the network layer by layer
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.25))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, n_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the classical network.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) or (batch_size, 784)

        Returns:
            Class logits of shape (batch_size, n_classes)
        """
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        return self.network(x)


class SimplifiedHybridQNN(nn.Module):
    """
    Simplified hybrid model for faster training with fewer classes.

    This model is designed for binary or 3-class classification tasks.
    It uses a simpler architecture that trains faster while still demonstrating
    quantum-classical hybrid computing.
    """

    def __init__(self, n_qubits=4, n_classes=2, feature_reps=2, var_reps=3):
        """
        Initialize the simplified hybrid network.

        Args:
            n_qubits: Number of qubits
            n_classes: Number of output classes (typically 2 or 3)
            feature_reps: Number of repetitions in the feature map
            var_reps: Number of repetitions in the variational circuit
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_classes = n_classes

        # Simpler preprocessing for faster computation
        self.pre_net = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits),
            nn.Tanh()
        )

        # Quantum layer
        self.quantum_layer = QuantumLayer(
            n_qubits=n_qubits,
            feature_reps=feature_reps,
            var_reps=var_reps
        )

        # Direct mapping from quantum output to classes
        self.post_net = nn.Linear(2, n_classes)

    def forward(self, x):
        """
        Forward pass through the simplified hybrid network.

        Args:
            x: Input tensor

        Returns:
            Class logits
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x = self.pre_net(x)
        x = self.quantum_layer(x)
        x = self.post_net(x)

        return x


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    This is useful for comparing model complexity and understanding
    computational requirements.

    Args:
        model: PyTorch model

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_size=(1, 1, 28, 28)):
    """
    Print a summary of the model architecture.

    Args:
        model: PyTorch model
        input_size: Size of input tensor for testing
    """
    print(f"\nModel: {model.__class__.__name__}")
    print("=" * 60)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("=" * 60)

    # Try to print layer information
    print("\nLayer structure:")
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            print(f"\n{name}:")
            for i, layer in enumerate(module):
                print(f"  ({i}): {layer}")
        else:
            print(f"{name}: {module.__class__.__name__}")


if __name__ == "__main__":
    # Demonstration of model creation
    print("Creating Hybrid Quantum-Classical Neural Network...")

    # Create a simplified hybrid model for demonstration
    try:
        model = SimplifiedHybridQNN(n_qubits=4, n_classes=2)
        model_summary(model)

        # Test forward pass with dummy data
        dummy_input = torch.randn(2, 784)  # Batch of 2 samples
        print("\nTesting forward pass with dummy input...")
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print("Model test successful!")

    except Exception as e:
        print(f"Error during model creation: {e}")
        print("This is expected if Qiskit components are not fully set up.")

    # Create classical baseline
    print("\n" + "=" * 60)
    print("Creating Classical Neural Network Baseline...")
    classical_model = ClassicalNN(n_classes=2)
    model_summary(classical_model)
