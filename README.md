# HKCH: Hypergraph Kolmogorov-Arnold Networks Contrastive Hashing for Unsupervised Cross-Modal Retrieval

[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official implementation of "Hypergraph Kolmogorov-Arnold Networks Contrastive Hashing for Unsupervised Cross-Modal Retrieval" - an unsupervised framework that integrates Fast-KANs with hypergraph neural networks for efficient cross-modal retrieval.

## Key Features

- **Fast-KANs Architecture**: Dynamically learnable spline-RBF activation functions for enhanced semantic mapping
- **Hypergraph Neural Networks**: Higher-order correlation modeling across modalities
- **Improved Ternary Contrastive Loss**: Mitigates false-negative pairs in unsupervised learning
- **Adaptive Hash Coding**: Dynamic weight adjustment for robust semantic alignment
- **Unsupervised Learning**: No requirement for labeled data

# Create conda environment
conda create -n hkch python=3.8
conda activate hkch

# Install dependencies
pip install torch==1.9.0 torchvision==0.10.0 torch-geometric
pip install scipy numpy tqdm