"""Phase 0 'done' criterion: the pinned ML stack is installed and works.

These tests fail (not skip) when torch / PyG are missing, so a broken environment is never
hidden. Run only these with:  pytest -m env
"""

import sys

import pytest

from src.check_env import check_pins, read_pins, smoke_test_graph_stack

pytestmark = pytest.mark.env


def test_python_311():
    assert sys.version_info[:2] == (3, 11), f"running {sys.version.split()[0]}"


def test_requirements_parse():
    pins = read_pins()
    assert {"torch", "torch_geometric", "lightgbm", "chromadb", "fastapi"} <= set(pins)
    assert "torch-geometric-temporal" not in pins  # replaced by src/models/stgnn.py


def test_pinned_versions_installed():
    problems = check_pins(read_pins())
    assert not problems, "\n".join(problems)


def test_graph_stack_imports_and_runs():
    out = smoke_test_graph_stack()
    assert out["GATConv"] == "(4, 16)"  # 2 heads x 8 channels, concatenated
    assert out["ChebConv"] == "(4, 8)"
