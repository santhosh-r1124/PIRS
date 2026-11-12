"""Cross-platform task runner: one command per stage, works on Windows without `make`.

    python run.py <stage> [extra args]

Examples:
    python run.py check-env
    python run.py test            # all tests
    python run.py test -k config  # extra args go to the last step (here: pytest)
    python run.py all             # whole pipeline, data -> models -> evaluate -> index

Another config: set SAP_CONFIG, e.g. SAP_CONFIG=configs/quick.yaml python run.py all
(Windows: set SAP_CONFIG=configs/quick.yaml, then python run.py all).

Each stage is a list of steps. A step names the source file it needs, the phase
that builds that file, and the command to run. If the file does not exist yet,
the runner says which phase adds it instead of failing with a stack trace.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# stage -> [(required path, phase that builds it, command), ...]
STAGES: dict[str, list[tuple[str, int, list[str]]]] = {
    "check-env": [("src/check_env.py", 0, [PY, "-m", "src.check_env"])],
    "data": [
        ("src/data/generate_synthetic.py", 1, [PY, "-m", "src.data.generate_synthetic"]),
        ("src/graph/sap_graph.py", 1, [PY, "-m", "src.graph.sap_graph"]),
        ("src/data/build_dataset.py", 2, [PY, "-m", "src.data.build_dataset"]),
    ],
    "train": [("src/train.py", 3, [PY, "-m", "src.train"])],
    "evaluate": [("src/evaluate.py", 6, [PY, "-m", "src.evaluate"])],
    "index": [
        ("src/recommender/build_index.py", 7, [PY, "-m", "src.recommender.build_index"]),
        ("src/recommender/recommend.py", 7, [PY, "-m", "src.recommender.recommend", "--evaluate"]),
    ],
    "api": [
        ("src/api/main.py", 8, [PY, "-m", "uvicorn", "src.api.main:app", "--port", "8000"]),
    ],
    "dashboard": [
        ("src/dashboard/app.py", 8, [PY, "-m", "streamlit", "run", "src/dashboard/app.py"]),
    ],
    "test": [("tests", 0, [PY, "-m", "pytest"])],
    "lint": [("pyproject.toml", 0, [PY, "-m", "ruff", "check", "."])],
}

# The Phase 6 ablations and rolling-origin folds (32 extra neural runs).
_TRAIN = [PY, "-m", "src.train", "--quiet"]
_GAT_COUNTS = ["--models", "gat_transformer", "--target", "regression"]
ABLATIONS: list[tuple[str, int, list[str]]] = [
    (
        "src/train.py",
        6,
        [
            *_TRAIN,
            "--models",
            "gat_transformer_none,gat_transformer_hand_built,gat_transformer_adaptive",
        ],
    ),
    ("src/train.py", 6, [*_TRAIN, *_GAT_COUNTS, "--lookback", "14"]),
    ("src/train.py", 6, [*_TRAIN, *_GAT_COUNTS, "--no-aux"]),
] + [
    (
        "src/train.py",
        6,
        [
            *_TRAIN,
            "--models",
            "lstm,gat_transformer",
            "--target",
            "regression",
            "--seeds",
            "42",
            "--split",
            f"fold{k}",
        ],
    )
    for k in range(1, 5)
]

# Whole pipeline from a clean clone: data -> every model -> ablations -> evaluate -> index.
# About 2 hours on a laptop CPU; with SAP_CONFIG=configs/quick.yaml a few minutes.
STAGES["all"] = [
    *STAGES["data"],
    ("src/train.py", 3, _TRAIN),
    *ABLATIONS,
    *STAGES["evaluate"],
    *STAGES["index"],
]


def run_stage(stage: str, extra_args: list[str]) -> int:
    steps = STAGES[stage]
    for i, (required, phase, cmd) in enumerate(steps):
        if not (ROOT / required).exists():
            print(f"[run.py] '{stage}': {required} is not built yet (added in Phase {phase}).")
            return 2
        if i == len(steps) - 1:
            cmd = cmd + extra_args
        print(f"[run.py] {' '.join(cmd[1:]) if cmd[0] == PY else ' '.join(cmd)}")
        code = subprocess.call(cmd, cwd=ROOT)
        if code != 0:
            return code
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        print(__doc__)
        print("Stages:", ", ".join(STAGES))
        return 1
    return run_stage(sys.argv[1], sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
