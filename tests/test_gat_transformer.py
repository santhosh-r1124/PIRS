"""Main model: shapes, information flows only along graph edges, attention / adjacency."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.graph.sap_graph import build_graph  # noqa: E402
from src.models.gat_transformer import GATTransformer  # noqa: E402
from src.utils import load_config  # noqa: E402

B, L, F, H, C = 3, 28, 16, 7, 5


@pytest.fixture(scope="module")
def graph():
    return build_graph(load_config())


def model(graph, mode):
    torch.manual_seed(0)
    return GATTransformer(
        graph.n,
        F,
        C,
        H,
        L,
        torch.as_tensor(graph.edge_index()),
        mode,
        d=16,
        heads=2,
        layers=2,
        ff=32,
        dropout=0.0,
    ).eval()


def inputs(n):
    torch.manual_seed(1)
    return torch.randn(B, L, n, F), torch.randn(B, H, C)


@pytest.mark.parametrize("mode", ["hand_built", "adaptive", "both", "none"])
def test_output_shape(graph, mode):
    x, xf = inputs(graph.n)
    assert model(graph, mode)(x, xf).shape == (B, H, graph.n)


def changed_modules(m, graph, module):
    """Which modules' forecasts change when one module's input history changes."""
    x, xf = inputs(graph.n)
    x2 = x.clone()
    x2[:, :, graph.index[module]] += 3.0
    with torch.no_grad():
        diff = (m(x, xf) - m(x2, xf)).abs().amax(dim=(0, 1))
    return {graph.modules[i] for i in range(graph.n) if diff[i] > 1e-6}


def test_information_follows_edge_direction(graph):
    """One GAT layer = one hop, in the edge direction: HCM -> FI, FI -> CO."""
    m = model(graph, "hand_built")
    assert changed_modules(m, graph, "HCM") == {"HCM", "FI"}
    assert changed_modules(m, graph, "FI") == {"FI", "CO"}  # not HCM: edge is HCM -> FI
    assert changed_modules(m, graph, "BASIS") == set(graph.modules)  # BASIS feeds everything


def test_no_graph_mode_keeps_modules_separate(graph):
    assert changed_modules(model(graph, "none"), graph, "SD") == {"SD"}


def test_attention_is_normalised_per_target(graph):
    m = model(graph, "hand_built")
    x, xf = inputs(graph.n)
    with torch.no_grad():
        m(x, xf, keep_attention=True)
    att = m.last_attention
    assert att["edges"].shape == (len(graph.edges), 2) and att["self"].shape == (graph.n, 2)
    # for every target module: incoming edges + self-loop sum to 1 (per head)
    total = att["self"].clone()
    dst = torch.as_tensor(graph.edge_index()[1])
    total.index_add_(0, dst, att["edges"])
    torch.testing.assert_close(total, torch.ones_like(total))


def test_learned_adjacency(graph):
    m = model(graph, "adaptive")
    a = m.learned_adjacency()
    assert a.shape == (graph.n, graph.n) and torch.all(a >= 0)
    torch.testing.assert_close(a.sum(0), torch.ones(graph.n))  # each target's weights sum to 1
    assert model(graph, "hand_built").learned_adjacency() is None
