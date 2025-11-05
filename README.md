# Quantum MNIST Classification

A hybrid quantum-classical neural network for classifying handwritten digits from the MNIST dataset. This project demonstrates how quantum computing can be integrated with classical machine learning using Qiskit and PyTorch.

## Overview

This project implements a hybrid architecture that combines classical neural networks with parameterized quantum circuits for image classification. The quantum component uses variational quantum circuits with trainable parameters, similar to weights in classical neural networks.

### Key Features

- Hybrid quantum-classical neural network architecture
- Binary and multi-class classification support
- Classical baseline model for performance comparison
- Comprehensive training and evaluation pipeline
- Detailed visualizations including training curves, confusion matrices, and ROC curves
- Modular and extensible code structure
- Interactive Jupyter notebook tutorial
- Works on free quantum simulators (no quantum hardware required)

## Architecture

The hybrid model consists of three main components:

1. **Classical Preprocessing**: Reduces the 784-dimensional MNIST images down to 4 features suitable for quantum processing
2. **Quantum Circuit**: A parameterized quantum circuit with 4 qubits that processes the compressed features
3. **Classical Postprocessing**: Maps the quantum circuit output to final class predictions

```
Input (28×28 image) → Classical Layers → Quantum Circuit → Classical Layers → Output
```

## Project Structure

```
Quantum_MNIST/
├── src/
│   ├── quantum_circuit.py      # Quantum circuit definitions
│   ├── models.py                # Quantum and classical model architectures
│   ├── data_utils.py            # Data loading and preprocessing
│   ├── train.py                 # Training utilities
│   └── visualize.py             # Evaluation and visualization tools
├── notebooks/
│   └── quantum_mnist_tutorial.ipynb  # Interactive tutorial
├── data/                        # MNIST dataset (auto-downloaded)
├── models/                      # Saved model checkpoints
├── results/
│   ├── plots/                   # Training curves, confusion matrices, etc.
│   └── metrics/                 # Performance metrics (CSV files)
├── logs/                        # Training logs
├── config.py                    # Configuration and hyperparameters
├── train_model.py              # Main training script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd Quantum_MNIST
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- qiskit (quantum computing framework)
- qiskit-machine-learning (quantum ML tools)
- torch and torchvision (deep learning)
- numpy, matplotlib, scikit-learn (scientific computing)
- jupyter (for notebooks)

## Quick Start

### Option 1: Run the Main Training Script

The easiest way to train both models and generate all results:

```bash
python train_model.py
```

This will:
- Download MNIST dataset (first run only)
- Train both quantum and classical models
- Generate training curves, confusion matrices, and ROC curves
- Save models and metrics to disk
- Print performance comparison

Expected runtime: 15-30 minutes depending on your hardware

### Option 2: Use the Jupyter Notebook

For a step-by-step walkthrough with detailed explanations:

```bash
jupyter notebook notebooks/quantum_mnist_tutorial.ipynb
```

The notebook provides:
- Interactive code cells with explanations
- Visualizations of quantum circuits
- Sample data exploration
- Model training and evaluation
- Experiments you can modify

## Configuration

Edit `config.py` to customize training parameters:

### Dataset Configuration

```python
# Binary classification (faster, recommended for testing)
DATASET_TYPE = 'binary'
BINARY_CLASS_A = 0        # First digit to classify
BINARY_CLASS_B = 1        # Second digit to classify
BINARY_TRAIN_SIZE = 500   # Samples per class

# Multi-class classification (slower but more comprehensive)
DATASET_TYPE = 'multiclass'
MULTICLASS_CLASSES = [0, 1, 2]  # Which digits to include
MULTICLASS_SAMPLES_PER_CLASS = 200
```

### Model Configuration

```python
MODEL_TYPE = 'hybrid'     # 'hybrid', 'classical', or 'both'
N_QUBITS = 4             # Number of qubits in quantum circuit
```

### Training Configuration

```python
LEARNING_RATE = 0.01
NUM_EPOCHS = 20
BATCH_SIZE = 32
```

## Usage Examples

### Train Only Quantum Model

Edit `config.py`:
```python
MODEL_TYPE = 'hybrid'
```

Then run:
```bash
python train_model.py
```

### Train Only Classical Baseline

Edit `config.py`:
```python
MODEL_TYPE = 'classical'
```

Then run:
```bash
python train_model.py
```

### Train and Compare Both Models

Edit `config.py`:
```python
MODEL_TYPE = 'both'
```

Then run:
```bash
python train_model.py
```

### Use a Saved Model

```python
import torch
from src.models import SimplifiedHybridQNN

# Load the model
model = SimplifiedHybridQNN(n_qubits=4, n_classes=2)
checkpoint = torch.load('models/quantum_hybrid_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Make predictions
model.eval()
predictions = model(test_images)
```

## Results

After training, you'll find:

### Saved Models
- `models/quantum_hybrid_best.pth` - Best quantum model checkpoint
- `models/classical_baseline_best.pth` - Best classical model checkpoint

### Visualizations
- `results/plots/quantum_hybrid_training_curves.png` - Training progress
- `results/plots/quantum_hybrid_confusion_matrix.png` - Classification errors
- `results/plots/quantum_hybrid_roc_curve.png` - ROC analysis
- `results/plots/model_comparison.png` - Side-by-side comparison

### Metrics
- `results/metrics/results.csv` - Quantitative performance metrics
- `logs/quantum_hybrid_history.json` - Detailed training history

## Expected Performance

For binary classification (0 vs 1) with default settings:

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| Quantum Hybrid | 85-95% | 20-30 min |
| Classical Baseline | 90-98% | 10-15 min |

Performance notes:
- Results vary based on random initialization
- Quantum simulations are computationally expensive
- Real quantum hardware would have different characteristics
- The goal is to demonstrate the approach, not achieve state-of-the-art accuracy

## Understanding the Quantum Component

### What are Parameterized Quantum Circuits?

The quantum circuits used in this project contain rotation gates with trainable parameters. During training, these parameters are optimized using gradient descent, just like weights in classical neural networks.

### Why Only 4 Qubits?

Current quantum simulators have practical limitations. Using 4 qubits:
- Keeps training time reasonable on CPU
- Demonstrates the concept effectively
- Scales to free cloud-based simulators

Real quantum computers could potentially use more qubits for larger problems.

### Feature Encoding

We use angle encoding to map classical data to quantum states:
- Classical values are normalized to [0, 2π]
- These angles parameterize rotation gates
- The resulting quantum state encodes the input features

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution**: The project works fine on CPU. Edit `config.py`:
```python
DEVICE = 'cpu'
```

### Issue: Slow training

**Solution**: Reduce the dataset size in `config.py`:
```python
BINARY_TRAIN_SIZE = 200  # Instead of 500
NUM_EPOCHS = 10          # Instead of 20
```

### Issue: Import errors

**Solution**: Make sure you activated the virtual environment and installed all dependencies:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: Quantum circuit errors

**Solution**: Ensure Qiskit is properly installed:
```bash
pip install --upgrade qiskit qiskit-machine-learning qiskit-aer
```

### Issue: Notebook kernel issues

**Solution**: Install the kernel in your virtual environment:
```bash
python -m ipykernel install --user --name=quantum_mnist
```

Then select this kernel in Jupyter.

## Extending the Project

### Add More Qubits

Edit `src/quantum_circuit.py` and `config.py` to increase the number of qubits. Note that training time increases exponentially.

### Try Different Quantum Circuits

Modify the ansatz in `src/quantum_circuit.py`:
```python
from qiskit.circuit.library import EfficientSU2

ansatz = EfficientSU2(num_qubits=4, reps=3)
```

### Use Real Quantum Hardware

Sign up for IBM Quantum and modify the code to use real quantum processors instead of simulators. See the Qiskit documentation for details.

### Multi-Class Classification

Change the configuration to classify more digits:
```python
DATASET_TYPE = 'multiclass'
MULTICLASS_CLASSES = [0, 1, 2, 3, 4]
```

## Technical Details

### Dependencies

- **Qiskit**: Open-source quantum computing framework
- **PyTorch**: Deep learning framework
- **Qiskit Machine Learning**: Quantum ML tools that bridge Qiskit and PyTorch
- **torchvision**: Provides MNIST dataset
- **matplotlib/seaborn**: Visualization
- **scikit-learn**: Metrics and evaluation

### Model Parameters

Quantum hybrid model (binary classification):
- Classical preprocessing: ~105,000 parameters
- Quantum circuit: 36 trainable parameters
- Classical postprocessing: ~50 parameters

The quantum circuit may have fewer parameters but can potentially represent complex functions due to the exponential state space of qubits.

## Learning Resources

### Quantum Computing Basics
- [IBM Quantum Learning](https://learning.quantum.ibm.com/)
- [Qiskit Textbook](https://qiskit.org/textbook/)

### Quantum Machine Learning
- [Qiskit Machine Learning Tutorials](https://qiskit.org/ecosystem/machine-learning/tutorials/)
- [PennyLane Quantum ML](https://pennylane.ai/qml/)

### Variational Quantum Algorithms
- [Variational Quantum Eigensolver](https://qiskit.org/textbook/ch-applications/vqe-molecules.html)
- [Quantum Neural Networks](https://arxiv.org/abs/1802.06002)

## Citation

If you use this code in your research, please cite:

```
@software{quantum_mnist_classifier,
  title = {Quantum MNIST Classification: Hybrid Quantum-Classical Neural Networks},
  year = {2024},
  url = {https://github.com/yourusername/quantum-mnist}
}
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for improvement:

- Add support for more quantum circuit designs
- Implement additional evaluation metrics
- Optimize quantum circuit depth
- Add support for other datasets
- Improve documentation

## Acknowledgments

- Built using Qiskit by IBM Research
- MNIST dataset by Yann LeCun
- Inspired by recent advances in quantum machine learning

## Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the maintainer.

## Future Work

- Implement quantum convolutional layers
- Explore quantum attention mechanisms
- Test on real quantum hardware
- Compare with other quantum ML approaches
- Extend to other image classification tasks

---

**Happy Quantum Computing!**
