"""SAP module graph: matrix conventions, PyG edge_index, leakage-free edge weights."""

import numpy as np
import pandas as pd
import pytest

from src.graph.sap_graph import SAPGraph, build_graph, cooccurrence_weights, plot_graph
from src.utils import load_config


@pytest.fixture(scope="module")
def graph():
    return build_graph(load_config())


def test_adjacency_direction(graph):
    a = graph.adjacency()
    i = graph.index
    assert a.shape == (10, 10)
    assert a[i["SD"], i["FI"]] == 1 and a[i["FI"], i["SD"]] == 0  # row = source
    assert np.all(np.diag(a) == 0)
    assert a.sum() == len(graph.edges)
    assert a[:, i["BASIS"]].sum() == 0 and a[i["BASIS"]].sum() == 9


def test_edge_index_matches_adjacency(graph):
    ei = graph.edge_index()
    assert ei.shape == (2, len(graph.edges)) and ei.dtype == np.int64
    rebuilt = np.zeros((graph.n, graph.n))
    rebuilt[ei[0], ei[1]] = 1
    np.testing.assert_array_equal(rebuilt, graph.adjacency())
    w = np.arange(100, dtype=float).reshape(10, 10)
    np.testing.assert_array_equal(graph.edge_weight(w), w[ei[0], ei[1]])


def test_invalid_edges_rejected():
    with pytest.raises(ValueError):
        SAPGraph(["FI", "SD"], [("SD", "XX")])
    with pytest.raises(ValueError):
        SAPGraph(["FI", "SD"], [("FI", "FI")])


def random_counts(graph, n_days=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D")
    return pd.DataFrame(rng.poisson(1.0, (n_days, graph.n)), index=idx, columns=graph.modules)


def test_cooccurrence_uses_training_period_only(graph):
    counts = random_counts(graph)
    train_end = counts.index[139]
    w1 = cooccurrence_weights(graph, counts, train_end)
    changed = counts.copy()
    changed.loc[changed.index > train_end] = 50  # wild values after the training period
    w2 = cooccurrence_weights(graph, changed, train_end)
    np.testing.assert_array_equal(w1, w2)
    assert w1.min() >= 0 and w1.max() == pytest.approx(1.0)
    assert np.all(w1[graph.adjacency() == 0] == 0)  # weights only on existing edges


def test_cooccurrence_finds_planted_ripple(graph):
    counts = random_counts(graph, n_days=400, seed=1)
    counts["FI"] = counts["SD"].shift(1, fill_value=0) * 2 + counts["FI"] // 2  # SD -> FI
    w = cooccurrence_weights(graph, counts, counts.index[-1])
    i = graph.index
    assert w[i["SD"], i["FI"]] == pytest.approx(1.0)


def test_plot_graph(graph, tmp_path):
    out = tmp_path / "g.png"
    plot_graph(graph, out)
    assert out.exists() and out.stat().st_size > 10_000
