"""Temporal-only deep models (no graph): LSTM and a small Transformer encoder.

Each module's look-back window is encoded separately with SHARED weights; a learned module
embedding tells the network which module it is looking at. Modules never exchange
information here - that is exactly what the graph models in Phase 5 add, so comparing the
two isolates the value of the graph.

Shapes:  x [B, L, N, F]  ->  encoding z [B, N, d]  ->  output [B, H, N]
The output is a log-rate for counts (Poisson) or a logit for P1/P2 (classification).
"""

from __future__ import annotations

import torch
from torch import nn


class ForecastHead(nn.Module):
    """Output for every (forecast day h, module) from the module's encoding, the calendar of
    the target day (known in advance) and a learned embedding of h."""

    def __init__(self, d: int, n_calendar: int, horizon: int, hidden: int = 64, h_emb: int = 8):
        super().__init__()
        self.h_emb = nn.Embedding(horizon, h_emb)
        self.mlp = nn.Sequential(
            nn.Linear(d + n_calendar + h_emb, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, z: torch.Tensor, xf: torch.Tensor) -> torch.Tensor:
        b, n, d = z.shape
        h = xf.shape[1]
        parts = [
            z[:, None].expand(b, h, n, d),
            xf[:, :, None].expand(b, h, n, xf.shape[-1]),
            self.h_emb.weight[None, :h, None].expand(b, h, n, -1),
        ]
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)  # [B, H, N]

    def init_bias(self, value: float) -> None:
        """Start from the training mean (log of mean count, or logit of the positive rate)."""
        nn.init.constant_(self.mlp[-1].bias, value)


class TemporalModel(nn.Module):
    """kind = "lstm" or "transformer"."""

    def __init__(
        self,
        kind: str,
        n_modules: int,
        n_features: int,
        n_calendar: int,
        horizon: int,
        max_lookback: int,
        d: int = 64,
        layers: int = 1,
        heads: int = 4,
        ff: int = 128,
        dropout: float = 0.1,
        module_emb: int = 8,
    ):
        super().__init__()
        if kind not in {"lstm", "transformer"}:
            raise ValueError(kind)
        self.kind = kind
        self.module_emb = nn.Embedding(n_modules, module_emb)
        self.inp = nn.Linear(n_features + module_emb, d)
        if kind == "lstm":
            self.encoder = nn.LSTM(
                d, d, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0
            )
        else:
            self.pos = nn.Parameter(torch.zeros(1, max_lookback, d))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = ForecastHead(d, n_calendar, horizon)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b, lb, n, _ = x.shape
        emb = self.module_emb.weight[None, None].expand(b, lb, n, -1)
        h = self.inp(torch.cat([x, emb], dim=-1))  # [B, L, N, d]
        h = h.permute(0, 2, 1, 3).reshape(b * n, lb, -1)  # one sequence per (sample, module)
        if self.kind == "lstm":
            out, _ = self.encoder(h)
        else:
            out = self.encoder(h + self.pos[:, -lb:])
        return out[:, -1].reshape(b, n, -1)  # state at the forecast origin

    def forward(self, x: torch.Tensor, xf: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x), xf)


def build_temporal(
    kind: str,
    cfg: dict,
    n_modules: int,
    n_features: int,
    n_calendar: int,
    horizon: int,
    max_lookback: int,
) -> TemporalModel:
    m = cfg["models"][kind]
    return TemporalModel(
        kind,
        n_modules,
        n_features,
        n_calendar,
        horizon,
        max_lookback,
        d=m["d_model"],
        layers=m["layers"],
        heads=m.get("heads", 4),
        ff=m.get("ff", 128),
        dropout=m["dropout"],
        module_emb=m["module_emb"],
    )
