"""Main model: Graph Attention (spatial) + Transformer (temporal), optional learned adjacency.

For every day in the look-back window, a graph layer lets each module read the same day's
features of the modules that can affect it; a Transformer then runs over each module's
sequence of daily embeddings; the shared ForecastHead (target-day calendar + horizon
embedding) predicts days 1..H.

graph modes
  hand_built  GAT over the SAP dependency graph from config.yaml
  adaptive    dense attention over a LEARNED adjacency A = softmax(relu(E_src E_tgt^T))
              (node-embedding adjacency as in Graph WaveNet; no prior graph needed)
  both        sum of the two branches
  none        no spatial layer at all (same network otherwise) - the "without graph" ablation

Every spatial branch is residual (x + graph(x)), so the model can learn to ignore the graph.
Shapes: x [B, L, N, F], xf [B, H, C] -> [B, H, N]
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv

from src.models.temporal import ForecastHead

GRAPH_MODES = {"hand_built", "adaptive", "both", "none"}


class AdaptiveAdjacency(nn.Module):
    """Learned adjacency over N nodes. weights()[j, i] = how much target i listens to source j
    (rows = sources, like the true propagation matrix); each column sums to 1."""

    def __init__(self, n_nodes: int, dim: int = 8):
        super().__init__()
        self.src = nn.Parameter(torch.randn(n_nodes, dim) * 0.1)
        self.tgt = nn.Parameter(torch.randn(n_nodes, dim) * 0.1)

    def weights(self) -> torch.Tensor:
        logits = torch.relu(self.src @ self.tgt.T)  # [source, target]
        return torch.softmax(logits, dim=0)  # normalise over sources for each target

    def forward(self, h: torch.Tensor, value: nn.Linear) -> torch.Tensor:
        """h [..., N, d] -> messages [..., N, d]: target i receives sum_j A[j, i] * value(h_j)."""
        return torch.einsum("ji,...jd->...id", self.weights(), value(h))


class GATTransformer(nn.Module):
    def __init__(
        self,
        n_modules: int,
        n_features: int,
        n_calendar: int,
        horizon: int,
        max_lookback: int,
        edge_index: torch.Tensor,
        graph: str = "hand_built",
        d: int = 64,
        heads: int = 4,
        layers: int = 2,
        ff: int = 128,
        dropout: float = 0.1,
        module_emb: int = 8,
        adj_dim: int = 8,
    ):
        super().__init__()
        if graph not in GRAPH_MODES:
            raise ValueError(f"graph must be one of {sorted(GRAPH_MODES)}")
        self.graph, self.n = graph, n_modules
        self.register_buffer("edge_index", edge_index.long(), persistent=False)
        self.module_emb = nn.Embedding(n_modules, module_emb)
        self.inp = nn.Linear(n_features + module_emb, d)
        if graph in {"hand_built", "both"}:
            self.gat = GATConv(d, d // heads, heads=heads, dropout=dropout, add_self_loops=True)
        if graph in {"adaptive", "both"}:
            self.adaptive = AdaptiveAdjacency(n_modules, adj_dim)
            self.adaptive_value = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.pos = nn.Parameter(torch.randn(1, max_lookback, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True)
        self.temporal = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = ForecastHead(d, n_calendar, horizon)
        self.last_attention: dict | None = None  # filled by forward(..., keep_attention=True)

    def _batched_edges(self, n_graphs: int) -> torch.Tensor:
        offsets = torch.arange(n_graphs, device=self.edge_index.device) * self.n
        return (self.edge_index[:, None, :] + offsets[None, :, None]).reshape(2, -1)

    def spatial(self, h: torch.Tensor, keep_attention: bool = False) -> torch.Tensor:
        """h [B, L, N, d] -> h + messages from other modules on the same day."""
        if self.graph == "none":
            return h
        b, lb, n, d = h.shape
        msg = torch.zeros_like(h)
        if self.graph in {"hand_built", "both"}:
            flat = h.reshape(b * lb * n, d)  # one small graph per (sample, day)
            out, (ei, alpha) = self.gat(
                flat, self._batched_edges(b * lb), return_attention_weights=True
            )
            msg = msg + out.reshape(b, lb, n, d)
            if keep_attention:
                # PyG returns [all G*E batched edges (graph by graph), then G*N self-loops
                # (node order)]. Average over the G = B*L small graphs: one weight per edge
                # and head. For each target, its in-edges + self-loop sum to 1 per head.
                g, e = b * lb, self.edge_index.shape[1]
                self.last_attention = {
                    "edges": alpha[: g * e].reshape(g, e, -1).mean(0).detach(),  # [E, heads]
                    "self": alpha[g * e :].reshape(g, n, -1).mean(0).detach(),  # [N, heads]
                }
        if self.graph in {"adaptive", "both"}:
            msg = msg + self.adaptive(h, self.adaptive_value)
        return self.norm(h + msg)

    def forward(self, x: torch.Tensor, xf: torch.Tensor, keep_attention: bool = False):
        b, lb, n, _ = x.shape
        emb = self.module_emb.weight[None, None].expand(b, lb, n, -1)
        h = self.inp(torch.cat([x, emb], dim=-1))  # [B, L, N, d]
        h = self.spatial(h, keep_attention)
        h = h.permute(0, 2, 1, 3).reshape(b * n, lb, -1) + self.pos[:, -lb:]
        z = self.temporal(h)[:, -1].reshape(b, n, -1)  # state at the forecast origin
        return self.head(z, xf)

    def learned_adjacency(self) -> torch.Tensor | None:
        """[source, target] learned weights (adaptive / both modes), else None."""
        return self.adaptive.weights().detach() if hasattr(self, "adaptive") else None


def build_gat_transformer(
    cfg: dict,
    n_modules: int,
    n_features: int,
    n_calendar: int,
    horizon: int,
    max_lookback: int,
    edge_index,
    graph: str | None = None,
) -> GATTransformer:
    m = cfg["models"]["gat_transformer"]
    return GATTransformer(
        n_modules,
        n_features,
        n_calendar,
        horizon,
        max_lookback,
        torch.as_tensor(edge_index),
        graph or m["graph"],
        d=m["d_model"],
        heads=m["heads"],
        layers=m["layers"],
        ff=m["ff"],
        dropout=m["dropout"],
        module_emb=m["module_emb"],
        adj_dim=m["adj_dim"],
    )
