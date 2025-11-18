# From Membership-Privacy Leakage to Quantum Machine Unlearning (QMU)

This repository provides the PyTorch + PennyLane implementation used in the paper:

> **From Membership-Privacy Leakage to Quantum Machine Unlearning**  
> Junjian Su, Runze He, Guanghui Li, Sujuan Qin, Zhimin He, Haozhen Situ, Fei Gao  

We study membership privacy leakage in Quantum Neural Networks (QNNs) and propose a **Quantum Machine Unlearning (QMU)** framework with three mechanisms (Gradient Ascent, Fisher-based, and Relative Gradient Ascent) to mitigate such leakage on both basic QNNs and Hybrid QNNs (HQNN).

This repository focuses on the **HQNN + Gradient Ascent Unlearning + MIA** pipeline used in our experiments on the 10-class MNIST dataset.

---

## Quick Start

```bash
python HQNN_original_train.py   # Train the original HQNN model
python HQNN_target_train.py     # Train the target HQNN model on retained data
python QMU_unlearning_GA.py     # Perform Gradient Ascent Unlearning (GA)
python MIA_attack.py            # Train and evaluate the MIA attack
---


For questions about the code or the paper, please contact: junjiansu2@gmail.com
