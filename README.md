# From Membership-Privacy Leakage to Quantum Machine Unlearning (QMU)

This repository provides the PyTorch + PennyLane implementation used in the paper:

> **From Membership-Privacy Leakage to Quantum Machine Unlearning**  
> Junjian Su, Runze He, Guanghui Li, Sujuan Qin, Zhimin He, Haozhen Situ, Fei Gao  

We study membership privacy leakage in Quantum Neural Networks (QNNs) and propose a **Quantum Machine Unlearning (QMU)** framework with three mechanisms (Gradient Ascent, Fisher-based, and Relative Gradient Ascent) to mitigate such leakage on both basic QNNs and Hybrid QNNs (HQNN).

> ⚠️ This repository focuses on the **HQNN + Gradient Ascent Unlearning + MIA** pipeline used in our experiments on the 10-class MNIST dataset.

---

## Environment

- Python ≥ 3.8  
- [PyTorch](https://pytorch.org/)  
- [torchvision](https://pytorch.org/vision/stable/index.html)  
- [PennyLane](https://pennylane.ai/)  
- NumPy  
- scikit-learn  

You can install the dependencies via:

```bash
pip install torch torchvision pennylane numpy scikit-learn



Repository Structure

The core scripts in this repository are:

HQNN_original_train.py
Trains the original HQNN model (denoted as Ao) on a subset of MNIST using a CNN pre-processing block followed by an 8-qubit hardware-efficient parameterized quantum circuit (PQC).
Saves:
model_original.pth (canonical name)
model_*.pth with a hyperparameter-based suffix
training curves in result/trained_params_*.pkl
