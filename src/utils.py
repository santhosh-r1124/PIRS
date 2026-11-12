"""Shared helpers used by every stage: config loading, paths and global seeding."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base` (lists and scalars are replaced)."""
    out = dict(base)
    for k, v in override.items():
        both_dicts = isinstance(v, dict) and isinstance(out.get(k), dict)
        out[k] = deep_merge(out[k], v) if both_dicts else v
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config.

    Default: $SAP_CONFIG if set, else configs/config.yaml. A config may start with
    `base: other.yaml` (relative to its own folder) to inherit everything from that file
    and override only the keys it lists; configs/quick.yaml does this.
    """
    p = Path(path or os.environ.get("SAP_CONFIG") or DEFAULT_CONFIG)
    if not p.is_absolute() and not p.exists():
        p = ROOT / p
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    return deep_merge(load_config(p.parent / base), cfg) if base else cfg


def get_path(cfg: dict[str, Any], key: str) -> Path:
    """Absolute path for an entry under `paths:` in the config; creates the folder."""
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every random number generator the project uses.

    Call once at the start of each script. For new code that needs randomness,
    prefer an explicit `np.random.default_rng(seed)` passed around as an argument,
    which keeps components independent of call order.
    """
    # Only affects subprocesses (the current interpreter's hash seed is fixed at start-up).
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Needed for deterministic cuBLAS kernels if a GPU is ever used; harmless on CPU.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # data-only stages can run without torch installed
        return

    torch.manual_seed(seed)  # also seeds all CUDA devices
    if deterministic:
        # warn_only: some scatter ops used by PyG have no deterministic CPU/GPU kernel;
        # we want a warning, not a crash, and we report results over 3 seeds anyway.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
