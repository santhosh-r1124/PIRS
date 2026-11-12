"""SAP module dependency graph (the "spatial" part of the spatio-temporal model).

Nodes are SAP modules; a directed edge A -> B means "a problem in A can ripple into B".
Conventions used everywhere in the project:
  adjacency[i, j] = weight of edge  modules[i] -> modules[j]   (row = source)
  edge_index      = [[sources], [targets]]  (PyTorch Geometric: messages flow source -> target,
                    so each module aggregates information from the modules that can affect it)

    python -m src.graph.sap_graph      # prints the graph and saves results/figures/sap_graph.png
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils import get_path, load_config  # noqa: E402


@dataclass
class SAPGraph:
    modules: list[str]
    edges: list[tuple[str, str]]

    def __post_init__(self):
        idx = self.index
        for a, b in self.edges:
            if a not in idx or b not in idx:
                raise ValueError(f"edge {a}->{b} uses a module not in {self.modules}")
            if a == b:
                raise ValueError(f"self-loop {a}->{b}")

    @property
    def index(self) -> dict[str, int]:
        return {m: i for i, m in enumerate(self.modules)}

    @property
    def n(self) -> int:
        return len(self.modules)

    def adjacency(self, weights: np.ndarray | None = None) -> np.ndarray:
        """[N, N] matrix, row = source. Binary unless `weights` (an [N, N] matrix) is given,
        in which case weights are kept only where an edge exists."""
        a = np.zeros((self.n, self.n), dtype=np.float32)
        for s, t in self.edges:
            a[self.index[s], self.index[t]] = 1.0
        if weights is not None:
            a = a * np.asarray(weights, dtype=np.float32)
        return a

    def edge_index(self) -> np.ndarray:
        """[2, E] int64 array for PyG: row 0 = source node, row 1 = target node."""
        return np.array(
            [[self.index[s] for s, _ in self.edges], [self.index[t] for _, t in self.edges]],
            dtype=np.int64,
        )

    def edge_weight(self, adjacency: np.ndarray) -> np.ndarray:
        """Edge weights in the same order as edge_index()."""
        src, dst = self.edge_index()
        return adjacency[src, dst].astype(np.float32)

    def to_networkx(self, adjacency: np.ndarray | None = None) -> nx.DiGraph:
        a = self.adjacency() if adjacency is None else adjacency
        g = nx.DiGraph()
        g.add_nodes_from(self.modules)
        for s, t in self.edges:
            g.add_edge(s, t, weight=float(a[self.index[s], self.index[t]]))
        return g


def build_graph(cfg: dict) -> SAPGraph:
    return SAPGraph(list(cfg["data"]["modules"]), [tuple(e) for e in cfg["graph"]["edges"]])


def cooccurrence_weights(
    graph: SAPGraph, daily_counts: pd.DataFrame, train_end: pd.Timestamp, max_lag: int = 3
) -> np.ndarray:
    """Edge weights from lagged co-occurrence in the TRAINING period only.

    For edge A -> B: Pearson correlation between A's count on day t and B's total count on
    days t+1..t+max_lag, using only days whose whole window ends on or before `train_end`
    (so no validation/test information leaks in). Negative values are clipped to 0 and the
    weights are scaled so the largest is 1. Shared seasonality (e.g. month-end close lifts
    FI, CO and SD together) also raises these correlations - a known confound.

    daily_counts: index = dates, columns = modules (incident counts per day).
    """
    train_end = pd.Timestamp(train_end)
    counts = daily_counts.sort_index()[graph.modules].astype(float)
    future = sum(counts.shift(-k) for k in range(1, max_lag + 1))  # B on t+1..t+lag
    last_ok = train_end - pd.Timedelta(days=max_lag)
    mask = (counts.index <= last_ok) & future.notna().all(axis=1)
    x, y = counts[mask], future[mask]

    w = np.zeros((graph.n, graph.n), dtype=np.float32)
    for s, t in graph.edges:
        i, j = graph.index[s], graph.index[t]
        xs, ys = x.iloc[:, i].to_numpy(), y.iloc[:, j].to_numpy()
        if xs.std() > 0 and ys.std() > 0:
            w[i, j] = max(0.0, float(np.corrcoef(xs, ys)[0, 1]))
    if w.max() > 0:
        w /= w.max()
    return graph.adjacency(w)


# Fixed layout: platform at the bottom, logistics in the middle, finance at the top.
LAYOUT = {
    "HCM": (-2.6, 2.0), "FI": (-0.9, 2.0), "CO": (1.3, 2.0),
    "SD": (-2.5, 0.7), "MM": (-0.1, 0.8), "PP": (2.6, 0.9),
    "WM": (-1.7, -0.5), "QM": (0.9, -0.3), "PM": (2.4, -0.7),
    "BASIS": (-0.4, -1.9),
}  # fmt: skip


def plot_graph(
    graph: SAPGraph, path, adjacency: np.ndarray | None = None, title: str | None = None
) -> None:
    """Draw the module graph; edge width follows the adjacency weights."""
    g = graph.to_networkx(adjacency)
    pos = {m: LAYOUT.get(m, (np.cos(i), np.sin(i))) for i, m in enumerate(graph.modules)}
    weights = np.array([d["weight"] for *_, d in g.edges(data=True)])
    widths = 0.6 + 2.4 * (weights / weights.max() if weights.max() > 0 else weights)
    basis_edge = ["BASIS" == s for s, _ in g.edges()]

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(
        g, pos, node_size=1500, node_color="#dbe7f3", edgecolors="#3b6ea5", ax=ax
    )
    nx.draw_networkx_labels(g, pos, font_size=10, font_weight="bold", ax=ax)
    functional = [e for e, b in zip(g.edges(), basis_edge, strict=True) if not b]
    platform = [e for e, b in zip(g.edges(), basis_edge, strict=True) if b]
    w = dict(zip(g.edges(), widths, strict=True))
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=functional,
        width=[w[e] for e in functional],
        arrowsize=14,
        node_size=1500,
        edge_color="#3b6ea5",
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )
    nx.draw_networkx_edges(
        g, pos, edgelist=platform, width=[0.6 * w[e] for e in platform], arrowsize=10,
        node_size=1500, edge_color="#a8a8a8", style="dashed", connectionstyle="arc3,rad=0.12",
        ax=ax,
    )  # fmt: skip
    default_title = (
        f"SAP module dependency graph: {graph.n} modules, {len(graph.edges)} directed edges\n"
        "arrow A → B = a problem in A can ripple into B; dashed grey = BASIS platform edges"
    )
    ax.set_title(title or default_title, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    graph = build_graph(cfg)
    a = graph.adjacency()
    print(f"{graph.n} modules, {len(graph.edges)} directed edges")
    print(pd.DataFrame(a.astype(int), index=graph.modules, columns=graph.modules).to_string())
    print("in-degree :", dict(zip(graph.modules, a.sum(0).astype(int), strict=True)))
    out = get_path(cfg, "figures") / "sap_graph.png"
    plot_graph(graph, out)
    print(f"saved {out.relative_to(get_path(cfg, 'results').parent)}")


if __name__ == "__main__":
    main()
