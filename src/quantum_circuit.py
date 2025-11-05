"""
Quantum Circuit Architecture for MNIST Classification

This module defines the parameterized quantum circuits used for feature extraction
and classification in the hybrid quantum-classical neural network.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes


class QuantumFeatureMap:
    """
    Creates quantum feature maps to encode classical data into quantum states.

    We use angle encoding to map normalized pixel values into rotation angles
    on the quantum circuit. This is a common approach for encoding classical
    data into quantum states.
    """

    def __init__(self, n_qubits=4, feature_dimension=4, reps=2):
        """
        Initialize the quantum feature map.

        Args:
            n_qubits: Number of qubits in the circuit
            feature_dimension: Dimension of input features (should match n_qubits)
            reps: Number of repetitions for the encoding circuit
        """
        self.n_qubits = n_qubits
        self.feature_dimension = feature_dimension
        self.reps = reps

    def create_feature_map(self):
        """
        Create a ZZ feature map circuit for data encoding.

        The ZZ feature map uses Hadamard gates followed by controlled-phase
        rotations to create entanglement between qubits based on input features.
        This helps capture non-linear relationships in the data.

        Returns:
            QuantumCircuit: The feature map circuit
        """
        feature_map = ZZFeatureMap(
            feature_dimension=self.feature_dimension,
            reps=self.reps,
            entanglement='linear'
        )
        return feature_map


class QuantumVariationalCircuit:
    """
    Creates the variational quantum circuit with trainable parameters.

    This circuit contains parameterized rotation gates that will be optimized
    during training. Think of these parameters as weights in a classical neural
    network - they are adjusted to minimize the loss function.
    """

    def __init__(self, n_qubits=4, reps=3):
        """
        Initialize the variational circuit.

        Args:
            n_qubits: Number of qubits in the circuit
            reps: Number of repetitions of the variational layer
        """
        self.n_qubits = n_qubits
        self.reps = reps

    def create_ansatz(self):
        """
        Create a RealAmplitudes ansatz as the variational form.

        RealAmplitudes uses RY rotation gates and controlled-X gates to create
        a flexible circuit that can represent a wide variety of quantum states.
        The parameters of the RY gates are what we'll train.

        Returns:
            QuantumCircuit: The variational circuit
        """
        ansatz = RealAmplitudes(
            num_qubits=self.n_qubits,
            reps=self.reps,
            entanglement='linear'
        )
        return ansatz


class HybridQuantumCircuit:
    """
    Combines feature map and variational circuit into a complete quantum circuit.

    This represents the full quantum part of our hybrid model. Input data flows
    through the feature map (encoding), then through the variational circuit
    (trainable quantum operations), and finally we measure to get classical output.
    """

    def __init__(self, n_qubits=4, feature_dim=4, feature_reps=2, var_reps=3):
        """
        Initialize the hybrid quantum circuit.

        Args:
            n_qubits: Number of qubits
            feature_dim: Dimension of input features
            feature_reps: Repetitions in feature map
            var_reps: Repetitions in variational circuit
        """
        self.n_qubits = n_qubits
        self.feature_dim = feature_dim
        self.feature_reps = feature_reps
        self.var_reps = var_reps

        # Create the component circuits
        self.feature_map = QuantumFeatureMap(n_qubits, feature_dim, feature_reps).create_feature_map()
        self.ansatz = QuantumVariationalCircuit(n_qubits, var_reps).create_ansatz()

    def create_circuit(self):
        """
        Combine feature map and ansatz into a complete circuit.

        The data flows like this:
        1. Classical input → Feature Map (encode into quantum state)
        2. Quantum state → Variational Circuit (apply trainable operations)
        3. Measure → Classical output

        Returns:
            QuantumCircuit: Complete quantum circuit for the model
        """
        # Combine the feature map and ansatz
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(self.feature_map, inplace=True)
        circuit.compose(self.ansatz, inplace=True)

        return circuit

    def get_num_parameters(self):
        """
        Calculate the total number of trainable parameters.

        This tells us how many weights we need to optimize during training.
        More parameters give more flexibility but also require more training data.

        Returns:
            int: Number of trainable parameters
        """
        return self.ansatz.num_parameters

    def draw_circuit(self, filename=None):
        """
        Visualize the quantum circuit.

        This helps us understand the structure of our quantum model.

        Args:
            filename: If provided, save the circuit diagram to this file

        Returns:
            matplotlib.figure.Figure: Circuit diagram
        """
        circuit = self.create_circuit()
        fig = circuit.draw(output='mpl', fold=20)

        if filename:
            fig.savefig(filename, dpi=300, bbox_inches='tight')

        return fig


def create_simple_circuit(n_qubits=4, n_layers=2):
    """
    Create a simple parameterized quantum circuit for testing.

    This is a more basic circuit that can be useful for quick experiments
    or debugging. It uses simple rotation gates without entanglement.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of layers of rotation gates

    Returns:
        QuantumCircuit: A simple parameterized circuit
    """
    qc = QuantumCircuit(n_qubits)

    # Create parameters for the rotation gates
    params = ParameterVector('θ', length=n_qubits * n_layers * 3)
    param_idx = 0

    # Apply rotation gates in layers
    for layer in range(n_layers):
        # Apply RX, RY, RZ rotations to each qubit
        for qubit in range(n_qubits):
            qc.rx(params[param_idx], qubit)
            param_idx += 1
            qc.ry(params[param_idx], qubit)
            param_idx += 1
            qc.rz(params[param_idx], qubit)
            param_idx += 1

        # Add entanglement between adjacent qubits
        if layer < n_layers - 1:
            for qubit in range(n_qubits - 1):
                qc.cx(qubit, qubit + 1)

    return qc


if __name__ == "__main__":
    # Demonstration of quantum circuit creation
    print("Creating Hybrid Quantum Circuit...")

    hybrid_circuit = HybridQuantumCircuit(
        n_qubits=4,
        feature_dim=4,
        feature_reps=2,
        var_reps=3
    )

    circuit = hybrid_circuit.create_circuit()
    print(f"Circuit created with {circuit.num_qubits} qubits")
    print(f"Number of trainable parameters: {hybrid_circuit.get_num_parameters()}")
    print(f"\nCircuit depth: {circuit.depth()}")
    print(f"Circuit size (gate count): {circuit.size()}")

    # Draw the circuit
    try:
        hybrid_circuit.draw_circuit('../results/plots/quantum_circuit.png')
        print("\nCircuit diagram saved to results/plots/quantum_circuit.png")
    except Exception as e:
        print(f"\nCould not save circuit diagram: {e}")
