"""One training loop for every neural model (temporal and graph models).

  loss        regression: Poisson negative log-likelihood on the log-rate output
              classification: binary cross-entropy, positives weighted by negatives/positives
  optimiser   Adam + weight decay; learning rate halved when validation loss stalls
  stability   gradient-norm clipping
  stopping    early stopping on validation loss; the best weights are restored
  resuming    optional per-epoch checkpoint + time budget (see fit)

Mini-batches are drawn in random order from the TRAINING windows only. That does not mix
time periods: validation and test windows are never seen during fitting.
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


def _tensor(a: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a), dtype=torch.float32)


class Objective:
    """Loss for a target type and the conversion from raw output to a prediction."""

    def __init__(self, target: str, y_train: np.ndarray, pos_weight: str | float = "auto"):
        self.target = target
        if target == "regression":
            self.bias = float(np.log(max(y_train.mean(), 1e-3)))
        else:
            p = float(np.clip(y_train.mean(), 1e-3, 1 - 1e-3))
            self.bias = float(np.log(p / (1 - p)))
            w = (1 - p) / p if pos_weight == "auto" else float(pos_weight)
            self.pos_weight = torch.tensor(w, dtype=torch.float32)

    def loss(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.target == "regression":
            return F.poisson_nll_loss(out.clamp(max=8.0), y, log_input=True, full=False)
        return F.binary_cross_entropy_with_logits(out, y, pos_weight=self.pos_weight)

    def predict(self, out: torch.Tensor) -> torch.Tensor:
        return torch.exp(out.clamp(max=8.0)) if self.target == "regression" else torch.sigmoid(out)


def _save_atomic(obj: dict, path: Path) -> None:
    """Write to a temporary file, then rename: an interrupted save never leaves a broken file."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def fit(
    model: nn.Module,
    data: dict,
    target: str,
    tcfg: dict,
    seed: int,
    verbose: bool = True,
    checkpoint: str | Path | None = None,
    time_budget: float | None = None,
) -> dict:
    """Train on data["train"], early-stop on data["val"]. Returns training statistics.

    Resumable: if `checkpoint` is given, the complete training state (weights, optimiser,
    scheduler, early-stopping counters and both random-number streams) is saved there after
    every epoch, and training continues from it if the file exists. With `time_budget`
    (seconds) training stops after the first epoch that ends past the budget and returns
    {"status": "paused"}; calling fit() again continues exactly where it stopped, giving the
    same result as one uninterrupted run (tested).
    """
    torch.manual_seed(seed)
    torch.set_num_threads(tcfg.get("threads", 2))
    tr, va = data["train"], data["val"]
    obj = Objective(target, tr["Y"], tcfg.get("pos_weight", "auto"))
    if hasattr(model, "head"):
        model.head.init_bias(obj.bias)

    ds = torch.utils.data.TensorDataset(_tensor(tr["X"]), _tensor(tr["Xf"]), _tensor(tr["Y"]))
    gen = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=tcfg["batch_size"], shuffle=True, generator=gen
    )
    xv, xfv, yv = _tensor(va["X"]), _tensor(va["Xf"]), _tensor(va["Y"])
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=tcfg["lr_factor"], patience=tcfg["lr_patience"]
    )

    st = {
        "epoch": 0,
        "best": float("inf"),
        "best_state": None,
        "best_epoch": 0,
        "wait": 0,
        "history": [],
        "elapsed": 0.0,
    }
    checkpoint = Path(checkpoint) if checkpoint else None
    if checkpoint and checkpoint.exists():
        ck = torch.load(checkpoint, weights_only=False)  # our own file, written below
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"])
        torch.set_rng_state(ck["rng"])
        st = ck["state"]
        if verbose:
            print(f"  resumed after epoch {st['epoch']}")

    start, elapsed_before, status = time.time(), st["elapsed"], "done"
    for epoch in range(st["epoch"] + 1, tcfg["max_epochs"] + 1):
        model.train()
        total, count = 0.0, 0
        for xb, xfb, yb in loader:
            opt.zero_grad()
            loss = obj.loss(model(xb, xfb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            total, count = total + loss.item() * len(xb), count + len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = obj.loss(model(xv, xfv), yv).item()
        sched.step(val_loss)
        st["history"].append(
            {
                "epoch": epoch,
                "train_loss": total / count,
                "val_loss": val_loss,
                "lr": opt.param_groups[0]["lr"],
            }
        )
        if val_loss < st["best"] - 1e-5:
            st.update(
                best=val_loss,
                best_state=copy.deepcopy(model.state_dict()),
                best_epoch=epoch,
                wait=0,
            )
        else:
            st["wait"] += 1
        st["epoch"], st["elapsed"] = epoch, elapsed_before + time.time() - start
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"  epoch {epoch:3d}  train {total / count:.4f}  val {val_loss:.4f}")
        finished = st["wait"] >= tcfg["patience"] or epoch == tcfg["max_epochs"]
        if checkpoint and not finished:
            _save_atomic(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "gen": gen.get_state(),
                    "rng": torch.get_rng_state(),
                    "state": st,
                },
                checkpoint,
            )
        if finished:
            break
        if time_budget is not None and time.time() - start > time_budget:
            status = "paused"
            break
    if status == "paused":
        return {"status": "paused", "epochs_run": st["epoch"]}
    model.load_state_dict(st["best_state"])
    return {
        "status": "done",
        "best_epoch": st["best_epoch"],
        "epochs_run": st["epoch"],
        "best_val_loss": st["best"],
        "train_seconds": round(st["elapsed"], 1),
        "history": st["history"],
        "n_params": sum(p.numel() for p in model.parameters()),
        "objective": obj,
    }


@torch.no_grad()
def predict(model: nn.Module, windows: dict, objective: Objective, batch: int = 256) -> np.ndarray:
    """[S, H, N] predictions (expected counts, or P1/P2 probabilities)."""
    model.eval()
    x, xf = _tensor(windows["X"]), _tensor(windows["Xf"])
    outs = [
        objective.predict(model(x[i : i + batch], xf[i : i + batch]))
        for i in range(0, len(x), batch)
    ]
    return torch.cat(outs).numpy()
