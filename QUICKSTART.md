# Quick Start Guide

Get your Quantum MNIST classifier running in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM

## Installation Steps

### 1. Navigate to Project Directory

```bash
cd Quantum_MNIST
```

### 2. Create Virtual Environment (Optional but Recommended)

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This takes 3-5 minutes and installs:
- Qiskit for quantum computing
- PyTorch for deep learning
- Matplotlib for visualization
- And other scientific computing tools

### 4. Run Your First Training

```bash
python train_model.py
```

This will:
- Download MNIST dataset automatically (first run only)
- Train a quantum hybrid model
- Train a classical baseline model
- Generate comparison plots
- Save results to `results/` directory

Expected runtime: 15-30 minutes

## What to Expect

### Console Output

You'll see:
```
============================================================
QUANTUM MNIST CLASSIFICATION
Hybrid Quantum-Classical Neural Network Training
============================================================

Loading MNIST data...
Binary classification: 0 vs 1

Training Quantum Hybrid Model...
Epoch 1/20 (45.2s):
  Train Loss: 0.4523 | Train Acc: 78.50%
  Val Loss:   0.4012 | Val Acc:   82.00%
  ...

Training Classical Baseline Model...
Epoch 1/20 (12.3s):
  Train Loss: 0.3891 | Train Acc: 85.25%
  Val Loss:   0.3445 | Val Acc:   88.50%
  ...

TRAINING COMPLETE!
Results Summary:
  Quantum Hybrid: 92.50% accuracy
  Classical: 95.00% accuracy
```

### Generated Files

After training, check these directories:

**models/** - Saved model checkpoints
- `quantum_hybrid_best.pth`
- `classical_baseline_best.pth`

**results/plots/** - Visualizations
- Training curves showing loss and accuracy over time
- Confusion matrices showing classification errors
- ROC curves showing model performance
- Comparison charts

**results/metrics/** - Performance data
- `results.csv` with accuracy, precision, recall, F1-score

**logs/** - Training history
- JSON files with detailed training logs

## Quick Configuration Changes

Want to try different settings? Edit `config.py`:

### Train Faster (for testing)

```python
BINARY_TRAIN_SIZE = 200    # Reduce from 500
NUM_EPOCHS = 10            # Reduce from 20
```

### Try Different Digits

```python
BINARY_CLASS_A = 3  # Instead of 0
BINARY_CLASS_B = 8  # Instead of 1
```

### Train Only Classical Model

```python
MODEL_TYPE = 'classical'  # Faster, for baseline testing
```

## Interactive Tutorial

For a step-by-step learning experience:

```bash
jupyter notebook notebooks/quantum_mnist_tutorial.ipynb
```

The notebook includes:
- Detailed explanations of quantum circuits
- Visualization of the architecture
- Sample predictions
- Interactive experiments

## Troubleshooting

### Error: "ModuleNotFoundError"

**Solution:** Make sure you installed dependencies:
```bash
pip install -r requirements.txt
```

### Error: "CUDA out of memory"

**Solution:** The project works on CPU. Edit `config.py`:
```python
DEVICE = 'cpu'
```

### Training is very slow

**Solution:** Quantum simulations are computationally intensive. To speed up:
1. Reduce training samples: `BINARY_TRAIN_SIZE = 200`
2. Reduce epochs: `NUM_EPOCHS = 10`
3. Train classical model only: `MODEL_TYPE = 'classical'`

### Import errors in Jupyter

**Solution:** Install Jupyter kernel for your virtual environment:
```bash
python -m ipykernel install --user --name=quantum_mnist
```
Then select this kernel in Jupyter.

## Next Steps

### 1. Examine Results

Look at the plots in `results/plots/`:
- How does training progress over time?
- Where does the model make mistakes?
- How does quantum compare to classical?

### 2. Experiment

Try modifications:
- Different digit pairs
- More/fewer training samples
- Adjust learning rate
- Change number of epochs

### 3. Read the Code

Start with:
1. `config.py` - See all available settings
2. `train_model.py` - Understand the training pipeline
3. `src/models.py` - See the model architectures
4. `src/quantum_circuit.py` - Understand quantum circuits

### 4. Extend the Project

Ideas:
- Add more quantum layers
- Try different quantum circuits
- Implement 10-class classification
- Use different datasets
- Add data augmentation

## Getting Help

1. Check `README.md` for detailed documentation
2. Review `PROJECT_SUMMARY.md` for technical details
3. Look at the troubleshooting section in README
4. Examine code comments for explanations

## Performance Tips

### For Fastest Training
```python
# In config.py
MODEL_TYPE = 'classical'        # Skip quantum training
BINARY_TRAIN_SIZE = 200         # Smaller dataset
NUM_EPOCHS = 10                 # Fewer epochs
```

### For Best Accuracy
```python
# In config.py
BINARY_TRAIN_SIZE = 1000        # More training data
NUM_EPOCHS = 30                 # More training iterations
LEARNING_RATE = 0.001           # Lower learning rate
```

### For Balanced Experience (Recommended)
```python
# In config.py (default settings)
MODEL_TYPE = 'both'             # Compare quantum and classical
BINARY_TRAIN_SIZE = 500         # Moderate dataset size
NUM_EPOCHS = 20                 # Good convergence
```

## What You'll Learn

By completing this quick start, you'll:

1. **Install and run** a quantum machine learning project
2. **Train models** using both quantum and classical approaches
3. **Evaluate performance** with comprehensive metrics
4. **Visualize results** with professional plots
5. **Compare** quantum and classical approaches

## Success Criteria

You're successful if you:
- [x] Installed all dependencies without errors
- [x] Ran `train_model.py` to completion
- [x] Generated plots in `results/plots/`
- [x] Saw model accuracy above 85%
- [x] Can open and view the generated visualizations

Congratulations! You've successfully trained a quantum-classical hybrid neural network.

## Time Commitment

- **Installation**: 5 minutes
- **First training run**: 20-30 minutes
- **Exploring results**: 10 minutes
- **Jupyter tutorial**: 30-60 minutes
- **Experimentation**: As much as you want!

---

**Ready to dive deeper?** Check out `README.md` for comprehensive documentation and advanced usage.
