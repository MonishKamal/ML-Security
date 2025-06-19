# Hands-On Labs in Machine Learning and Cybersecurity

## Overview

This project focuses on exploring **machine learning** and **cybersecurity**, specifically emphasizing adversarial attacks and federated learning. It provides hands-on labs to bridge theoretical knowledge with practical implementation in AI-driven threat mitigation.

### Key Objectives

1. **Understanding ML Security**:

   - Analyze vulnerabilities in machine learning models.
   - Devise strategies for mitigating attacks.

2. **Adversarial Training**:

   - Enhance model robustness through adversarial examples.
   - Evaluate defense mechanisms like FGSM, PGD, and BIM.

3. **Federated Learning**:
   - Investigate decentralized learning approaches for enhanced privacy and security.

---

## Project Highlights

- **Adversarial Attacks**:

  - **FGSM (Fast Gradient Sign Method)**: Craft perturbations to mislead models.
  - **PGD (Projected Gradient Descent)**: Iterative perturbations for stronger adversarial examples.
  - **CW Attack**: Minimized distortion while ensuring misclassification.
  - **BIM (Basic Iterative Method)**: Iterative FGSM for refined attacks.

- **Datasets**:

  - **CIFAR-10**: 60,000 color images across 10 classes for benchmarking.
  - **Fashion-MNIST**: 70,000 grayscale images of clothing items for diverse model evaluation.

- **Federated Learning Architecture**:
  - Simulate federated environments with normal and adversarial clients.
  - Secure communication protocols for decentralized model training.

---

## Experimental Setup

- **Models**:

  - Convolutional Neural Networks (CNN)
  - ResNet-18 with residual learning for robustness.

- **Training Parameters**:

  - Learning Rate: `0.01`
  - Batch Size: `64`
  - Optimization Algorithm: SGD with momentum.

- **Evaluation Metrics**:

  - **Accuracy**: Performance on clean and adversarial datasets.
  - **Adversarial Robustness**: Resistance to perturbations.

- **Hardware**:
  - NVIDIA Quadro RTX 5000 GPU for model training.

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- Additional libraries: `numpy`, `matplotlib`, `scikit-learn`, `cleverhans`

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/tianpu2014/MLSecurity.git
   ```
