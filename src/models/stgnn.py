"""Spatio-temporal GNNs from the literature, re-implemented on PyTorch Geometric / PyTorch.

Why re-implemented: PyTorch Geometric Temporal (which ships these models) needs the
torch-sparse C++ extension, which could not be installed in the environment used for this
project (see docs/phase_log.md, Phase 5). These are the closest equivalents, kept small:

  STGCN (Yu et al., 2018): two ST-Conv blocks, each = gated temporal convolution (GLU) ->
      Chebyshev graph convolution (PyG ChebConv) -> gated temporal convolution -> LayerNorm
      (per module; see STConvBlock). Spectral graph convolution assumes a symmetric
      Laplacian, so it uses the UNDIRECTED version of the SAP graph.
  DCRNN (Li et al., 2018): a GRU whose matrix products are diffusion convolutions over the
      DIRECTED graph - K random-walk steps forward along the edges (source -> target) and K
      backward. Simplification: encoder only; the original seq2seq decoder with scheduled
      sampling is replaced by the same direct multi-horizon head every other model uses.

Both end in the shared ForecastHead (target-day calendar + horizon embedding), so results
differ only by the spatio-temporal encoder.  Shapes: x [B, L, N, F], xf [B, H, C] -> [B, H, N]
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import ChebConv
from torch_geometric.utils import to_undirected

from src.models.temporal import ForecastHead


def batch_edges(edge_index: torch.Tensor, n_graphs: int, n_nodes: int) -> torch.Tensor:
    """The same small graph repeated n_graphs times, as one PyG edge_index."""
    offsets = torch.arange(n_graphs, device=edge_index.device) * n_nodes
    return (edge_index[:, None, :] + offsets[None, :, None]).reshape(2, -1)


# --------------------------------------------------------------------------- STGCN
class GatedTemporalConv(nn.Module):
    """GLU over time for every node: (P + residual) * sigmoid(Q); shortens T by kt - 1."""

    def __init__(self, c_in: int, c_out: int, kt: int):
        super().__init__()
        self.kt = kt
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(kt, 1))
        self.res = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x [B, C, T, N]
        p, q = self.conv(x).chunk(2, dim=1)
        return (p + self.res(x)[:, :, self.kt - 1 :]) * torch.sigmoid(q)


class STConvBlock(nn.Module):
    def __init__(self, d: int, kt: int, cheb_k: int, dropout: float):
        super().__init__()
        self.t1 = GatedTemporalConv(d, d, kt)
        self.graph = ChebConv(d, d, K=cheb_k, normalization="sym")
        self.t2 = GatedTemporalConv(d, d, kt)
        # Deliberate deviation from reference code, which normalises over [nodes, channels]
        # jointly: that shares statistics between modules, i.e. information flows between
        # modules OUTSIDE the graph (caught by tests/test_stgnn.py). Normalise per module.
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.t1(x)  # [B, d, T', N]
        b, d, t, n = h.shape
        flat = h.permute(0, 2, 3, 1).reshape(b * t * n, d)  # one graph per (sample, day)
        g = torch.relu(self.graph(flat, batch_edges(edge_index, b * t, n)))
        h = self.t2(g.reshape(b, t, n, d).permute(0, 3, 1, 2))
        return self.drop(self.norm(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))


class STGCN(nn.Module):
    def __init__(
        self,
        n_modules: int,
        n_features: int,
        n_calendar: int,
        horizon: int,
        edge_index: torch.Tensor,
        d: int = 64,
        kt: int = 3,
        cheb_k: int = 3,
        blocks: int = 2,
        dropout: float = 0.1,
        module_emb: int = 8,
    ):
        super().__init__()
        self.register_buffer("edge_index", to_undirected(edge_index.long()), persistent=False)
        self.module_emb = nn.Embedding(n_modules, module_emb)
        self.inp = nn.Linear(n_features + module_emb, d)
        self.blocks = nn.ModuleList(STConvBlock(d, kt, cheb_k, dropout) for _ in range(blocks))
        self.out = nn.Linear(2 * d, d)  # [last step, mean over remaining steps] -> d
        self.head = ForecastHead(d, n_calendar, horizon)

    def forward(self, x: torch.Tensor, xf: torch.Tensor) -> torch.Tensor:
        b, lb, n, _ = x.shape
        emb = self.module_emb.weight[None, None].expand(b, lb, n, -1)
        h = self.inp(torch.cat([x, emb], dim=-1)).permute(0, 3, 1, 2)  # [B, d, L, N]
        for block in self.blocks:
            h = block(h, self.edge_index)
        z = torch.cat([h[:, :, -1], h.mean(dim=2)], dim=1).permute(0, 2, 1)  # [B, N, 2d]
        return self.head(torch.relu(self.out(z)), xf)


# --------------------------------------------------------------------------- DCRNN
def diffusion_matrices(edge_index: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random-walk transition matrices for message passing on a directed graph.

    forward[i, j]  = A[j, i] / in-degree(i)   target i averages its sources j (along edges)
    backward[i, j] = A[i, j] / out-degree(i)  source i averages its targets j (against edges)
    Rows of nodes without such neighbours are zero (e.g. BASIS has no sources)."""
    a = torch.zeros(n, n)
    a[edge_index[0], edge_index[1]] = 1.0  # a[source, target]
    fwd = a.T / a.T.sum(1, keepdim=True).clamp(min=1)
    bwd = a / a.sum(1, keepdim=True).clamp(min=1)
    return fwd, bwd


class DiffusionConv(nn.Module):
    """Linear layer over [X, P_f X, ..., P_f^K X, P_b X, ..., P_b^K X] (node axis = -2)."""

    def __init__(self, c_in: int, c_out: int, k: int):
        super().__init__()
        self.k = k
        self.lin = nn.Linear(c_in * (2 * k + 1), c_out)

    def forward(self, x: torch.Tensor, fwd: torch.Tensor, bwd: torch.Tensor) -> torch.Tensor:
        terms = [x]
        for p in (fwd, bwd):
            h = x
            for _ in range(self.k):
                h = p @ h
                terms.append(h)
        return self.lin(torch.cat(terms, dim=-1))


class DCGRUCell(nn.Module):
    def __init__(self, c_in: int, d: int, k: int):
        super().__init__()
        self.gates = DiffusionConv(c_in + d, 2 * d, k)
        self.cand = DiffusionConv(c_in + d, d, k)

    def forward(self, x, h, fwd, bwd):  # x [B, N, c_in], h [B, N, d]
        r, u = torch.sigmoid(self.gates(torch.cat([x, h], -1), fwd, bwd)).chunk(2, dim=-1)
        c = torch.tanh(self.cand(torch.cat([x, r * h], -1), fwd, bwd))
        return u * h + (1 - u) * c


class DCRNN(nn.Module):
    def __init__(
        self,
        n_modules: int,
        n_features: int,
        n_calendar: int,
        horizon: int,
        edge_index: torch.Tensor,
        d: int = 64,
        diffusion_k: int = 2,
        dropout: float = 0.1,
        module_emb: int = 8,
    ):
        super().__init__()
        fwd, bwd = diffusion_matrices(edge_index.long(), n_modules)
        self.register_buffer("fwd", fwd, persistent=False)
        self.register_buffer("bwd", bwd, persistent=False)
        self.d = d
        self.module_emb = nn.Embedding(n_modules, module_emb)
        self.cell = DCGRUCell(n_features + module_emb, d, diffusion_k)
        self.drop = nn.Dropout(dropout)
        self.head = ForecastHead(d, n_calendar, horizon)

    def forward(self, x: torch.Tensor, xf: torch.Tensor) -> torch.Tensor:
        b, lb, n, _ = x.shape
        emb = self.module_emb.weight[None].expand(b, n, -1)
        h = x.new_zeros(b, n, self.d)
        for t in range(lb):  # recurrent over the look-back window
            h = self.cell(torch.cat([x[:, t], emb], -1), h, self.fwd, self.bwd)
        return self.head(self.drop(h), xf)


def build_stgnn(
    name: str, cfg: dict, n_modules: int, n_features: int, n_calendar: int, horizon: int, edge_index
) -> nn.Module:
    m, ei = cfg["models"][name], torch.as_tensor(edge_index)
    if name == "stgcn":
        return STGCN(
            n_modules,
            n_features,
            n_calendar,
            horizon,
            ei,
            d=m["d_model"],
            kt=m["kt"],
            cheb_k=m["cheb_k"],
            blocks=m["blocks"],
            dropout=m["dropout"],
            module_emb=m["module_emb"],
        )
    if name == "dcrnn":
        return DCRNN(
            n_modules,
            n_features,
            n_calendar,
            horizon,
            ei,
            d=m["d_model"],
            diffusion_k=m["diffusion_k"],
            dropout=m["dropout"],
            module_emb=m["module_emb"],
        )
    raise ValueError(name)
