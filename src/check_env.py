"""Environment check (Phase 0 "done" criterion).

    python run.py check-env

1. Python is 3.11.
2. Every `name==version` pin in requirements.txt is installed at exactly that version.
3. torch and torch_geometric import, and a tiny forward pass runs through the two PyG
   layers the models use (GATConv in the GAT-Transformer, ChebConv in STGCN).
Exit code 0 only if everything passes.
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([^\s#;]+)")


def read_pins(path: Path = ROOT / "requirements.txt") -> dict[str, str]:
    """{package: pinned version} for every `name==version` line."""
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = PIN.match(line)
        if m:
            pins[m.group(1)] = m.group(2)
    return pins


def check_pins(pins: dict[str, str]) -> list[str]:
    """Return one message per package that is missing or at the wrong version."""
    problems = []
    for name, wanted in pins.items():
        try:
            got = version(name)
        except PackageNotFoundError:
            problems.append(f"{name}: not installed (want {wanted})")
            continue
        if got.split("+")[0] != wanted:  # ignore local tags such as 2.6.0+cpu
            problems.append(f"{name}: installed {got}, pinned {wanted}")
    return problems


def smoke_test_graph_stack() -> dict[str, str]:
    """Import the deep-learning stack and run one tiny forward pass per layer type."""
    import torch
    import torch_geometric
    from torch_geometric.nn import ChebConv, GATConv

    torch.manual_seed(0)
    n_nodes, n_feat = 4, 3
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])  # a 4-node ring
    x = torch.randn(n_nodes, n_feat)
    out = {"torch": torch.__version__, "torch_geometric": torch_geometric.__version__}
    out["GATConv"] = str(tuple(GATConv(n_feat, 8, heads=2)(x, edge_index).shape))
    out["ChebConv"] = str(tuple(ChebConv(n_feat, 8, K=3)(x, edge_index).shape))
    return out


def main() -> int:
    ok = True
    py = sys.version_info
    print(f"Python {py.major}.{py.minor}.{py.micro} ({sys.executable})")
    if (py.major, py.minor) != (3, 11):
        print("  WARNING: the project targets Python 3.11; versions may not resolve elsewhere.")

    problems = check_pins(read_pins())
    print(f"\nPinned packages: {'all match' if not problems else f'{len(problems)} problem(s)'}")
    for p in problems:
        print(f"  - {p}")
    ok &= not problems

    print("\nDeep-learning stack smoke test:")
    try:
        for k, v in smoke_test_graph_stack().items():
            print(f"  {k:<26} {v}")
    except Exception as e:  # report any import/runtime failure instead of a stack trace
        print(f"  FAILED: {type(e).__name__}: {e}")
        ok = False

    print("\nRESULT:", "OK" if ok else "NOT OK")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
