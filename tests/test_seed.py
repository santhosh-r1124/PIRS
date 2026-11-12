"""Same seed -> same random numbers (reproducibility rule)."""

import random

import numpy as np
import pytest

from src.utils import set_seed


def _draw():
    return random.random(), np.random.rand(3).tolist()


def test_same_seed_same_numbers():
    set_seed(123)
    a = _draw()
    set_seed(123)
    assert _draw() == a


def test_different_seed_different_numbers():
    set_seed(1)
    a = _draw()
    set_seed(2)
    assert _draw() != a


def test_torch_is_seeded():
    torch = pytest.importorskip("torch")  # covered for real by test_environment.py
    set_seed(7)
    a = torch.randn(5)
    set_seed(7)
    assert torch.equal(a, torch.randn(5))
