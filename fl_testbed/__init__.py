"""
FL Security Testbed
===================
Byzantine attack and defense research framework built on Flower (flwr).

Research Question:
    Under realistic threat models (partial Byzantine fraction f/n ≤ 0.3),
    which aggregation defenses maintain accuracy above 85% on MNIST/CIFAR-10
    while keeping per-round computation overhead under 2× FedAvg?

Modules:
    - models     : PyTorch CNN baselines
    - data       : IID / non-IID data partitioning (Dirichlet α)
    - attacks    : Byzantine strategies (label-flip, gradient poisoning, backdoor)
    - defenses   : Robust aggregators (FedAvg, Krum, TrimmedMean, FLAME)
    - simulation : Flower simulation harness
    - runner     : CLI experiment runner with JSON logging
"""

__version__ = "0.1.0"
