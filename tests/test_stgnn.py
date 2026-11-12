"""STGCN / DCRNN equivalents: shapes, use of the graph, diffusion matrices."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.graph.sap_graph import build_graph  # noqa: E402
from src.models.stgnn import DCRNN, STGCN, diffusion_matrices  # noqa: E402
from src.utils import load_config  # noqa: E402

B, F, H, C = 3, 16, 7, 5


@pytest.fixture(scope="module")
def graph():
    return build_graph(load_config())


def make(kind, n, edge_index):
    torch.manual_seed(0)
    cls = STGCN if kind == "stgcn" else DCRNN
    return cls(n, F, C, H, torch.as_tensor(edge_index), d=16, dropout=0.0).eval()


def changed(m, n, module_idx, lookback=28):
    torch.manual_seed(1)
    x, xf = torch.randn(B, lookback, n, F), torch.randn(B, H, C)
    x2 = x.clone()
    x2[:, :, module_idx] += 3.0
    with torch.no_grad():
        diff = (m(x, xf) - m(x2, xf)).abs().amax(dim=(0, 1))
    return {i for i in range(n) if diff[i] > 1e-6}


@pytest.mark.parametrize("kind", ["stgcn", "dcrnn"])
@pytest.mark.parametrize("lookback", [28, 14])
def test_output_shape(graph, kind, lookback):
    m = make(kind, graph.n, graph.edge_index())
    out = m(torch.randn(B, lookback, graph.n, F), torch.randn(B, H, C))
    assert out.shape == (B, H, graph.n)


@pytest.mark.parametrize("kind", ["stgcn", "dcrnn"])
def test_graph_is_used(graph, kind):
    """A module's input reaches its neighbours; with no edges at all, nothing crosses over."""
    i = graph.index
    m = make(kind, graph.n, graph.edge_index())
    assert {i["SD"], i["FI"]} <= changed(m, graph.n, i["SD"])  # SD -> FI is an edge
    empty = torch.zeros((2, 0), dtype=torch.long)
    assert changed(make(kind, graph.n, empty), graph.n, i["SD"]) == {i["SD"]}


def test_diffusion_matrices(graph):
    fwd, bwd = diffusion_matrices(torch.as_tensor(graph.edge_index()), graph.n)
    i = graph.index
    assert fwd[i["FI"], i["SD"]] > 0 and fwd[i["SD"], i["FI"]] == 0  # FI listens to SD
    assert bwd[i["SD"], i["FI"]] > 0  # backward: SD listens to its target FI
    assert fwd[i["BASIS"]].sum() == 0  # BASIS has no sources
    has_sources = fwd.sum(1) > 0
    torch.testing.assert_close(fwd.sum(1)[has_sources], torch.ones(int(has_sources.sum())))
