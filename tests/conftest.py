"""pytest configuration — shared fixtures."""
import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def seed_rng():
    """Seed all RNGs before each test for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    yield
