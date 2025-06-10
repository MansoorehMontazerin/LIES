*** Sparse Interpretable Deep Learning with LIES Networks for Symbolic Regression ***

This project implements the LIES architecture for symbolic regression.
LIES uses a fixed deep network with interpretable activations (Logarithm, Identity, Exponential, Sine), structured pruning, simplification and coefficient optimization to discover compact symbolic expressions from data.

Usage:
- Run `LIES_pipeline.py` to train the whole pipeline.
- Use `extract.py` to retrieve symbolic expressions.


Main Features:
- Pruning with ADMM
- Node and weight sparsity enforcement
- Gradient-based rounding and coefficient optimization
- Supports AI Feynman dataset

Author: Anonymous (for double-blind review)


