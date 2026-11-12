"""Train / run forecasting models and save their metrics to results/.

    python -m src.train                                  # all models, all seeds, both targets
    python -m src.train --models baselines               # Phase 3 only
    python -m src.train --models lstm --target regression --seeds 42
    python -m src.train --summary                        # only rebuild the comparison tables
    python -m src.train --models gat_transformer --seeds 42 --time-budget 150
                                          # stop after ~150 s; rerun the same command to resume

Every neural run (model, target, seed) is saved on its own in results/runs/, so runs can be
added one at a time. The comparison tables (test split, mean ± std over seeds) are rebuilt
from whatever runs exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.build_dataset import (
    DailyData,
    Split,
    chronological_split,
    load_daily,
    prepare,
    rolling_origin_splits,
)
from src.metrics import score
from src.models.baselines import run_baselines
from src.utils import get_path, load_config, set_seed

NEURAL = ["lstm", "transformer", "stgcn", "dcrnn", "gat_transformer"]
# Phase 6 ablations of the main model: same network, graph mode forced
GAT_VARIANTS = [f"gat_transformer_{m}" for m in ["hand_built", "adaptive", "none"]]
BASELINES = ["naive", "seasonal_naive", "historical_mean", "lightgbm"]
LOWER_IS_BETTER = {"MAE", "RMSE", "PoissonDev"}
REPORTED = {
    "regression": ["MAE", "RMSE", "PoissonDev", "P@3"],
    "classification": ["PR_AUC", "ROC_AUC", "P@3", "R@3"],
}
NOTES = [
    "Compare models on the TEST split.",
    "LightGBM and the neural models pick their stopping point on the validation split, so "
    "their validation scores are slightly optimistic.",
    "P@3 (regression): share of the 3 highest-forecast modules that had >= 1 incident. A fixed "
    "ranking of the busiest modules already scores well here; read it against historical_mean.",
    "P@3 (classification): share of the 3 highest-scored modules with >= 1 P1/P2.",
    "R@3: share of all P1/P2 module-days among that day's 3 flagged modules (recall under a "
    "3-alerts-per-day budget; no threshold needed).",
    "recall/precision_at_threshold (JSON only) use the F1-optimal threshold chosen on validation.",
]


# --------------------------------------------------------------------------- tables
def metrics_table(metrics: dict, target: str, horizons: list[int]) -> pd.DataFrame:
    """Rows = (split, model); columns = metric@h."""
    rows = []
    for part, by_model in metrics.items():
        for model, m in by_model.items():
            row = {"split": part, "model": model}
            for name in REPORTED[target]:
                for h in horizons:
                    row[f"{name}@{h}"] = m[f"h{h}"][name]
            rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    """Small markdown table writer (avoids an extra dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- baselines
def run_baseline_stage(
    cfg: dict, daily: DailyData, split: Split, targets: list[str], seed: int
) -> None:
    horizons = cfg["dataset"]["horizons"]
    results_dir = get_path(cfg, "results")
    pred_dir = results_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    label = "SYNTHETIC DATA" if daily.synthetic else "REAL DATA"
    path = results_dir / "baselines.json"
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    out.update(
        {
            "data": label,
            "synthetic": daily.synthetic,
            "seed": seed,
            "split": split.describe(daily.dates),
            "horizons": horizons,
            "notes": NOTES,
        }
    )
    for target in targets:
        res = run_baselines(cfg, daily, split, target, seed)
        out[target] = res["metrics"]
        out[f"lightgbm_{target}"] = res["lightgbm"]
        arrays = {f"{m}__{p}": a for m, by_p in res["preds"].items() for p, a in by_p.items()}
        arrays.update({f"truth__{p}": a for p, a in res["truth"].items()})
        np.savez_compressed(
            pred_dir / f"baselines_{target}.npz",
            val_origins=split.val,
            test_origins=split.test,
            **arrays,
        )
        if res["importance"] is not None:
            res["importance"].to_csv(results_dir / f"lightgbm_importance_{target}.csv", index=False)
        table = metrics_table(res["metrics"], target, horizons)
        table.to_csv(results_dir / f"baselines_{target}.csv", index=False)
        print(f"\nbaselines ({target}) done; LightGBM: {res['lightgbm']['status']}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    md = [f"# Baselines ({label}) - validation and test splits", ""]
    for target in [t for t in ["regression", "classification"] if t in out]:
        table = metrics_table(out[target], target, horizons)
        for part in ["val", "test"]:
            md += [
                f"## {target} - {part}",
                "",
                to_markdown(table[table["split"] == part].drop(columns="split")),
                "",
            ]
        md += [f"LightGBM: {out[f'lightgbm_{target}']['status']}", ""]
    (results_dir / "baselines_table.md").write_text("\n".join(md), encoding="utf-8")


# --------------------------------------------------------------------------- neural models
def build_model(name: str, cfg: dict, daily: DailyData, data: dict):
    n_feat, n_cal = data["train"]["X"].shape[-1], data["train"]["Xf"].shape[-1]
    max_lb = max(cfg["dataset"]["lookback"], data["lookback"])
    if name in {"lstm", "transformer"}:
        from src.models.temporal import build_temporal

        return build_temporal(name, cfg, len(daily.modules), n_feat, n_cal, data["horizon"], max_lb)
    if name.startswith("gat_transformer"):
        from src.graph.sap_graph import build_graph
        from src.models.gat_transformer import build_gat_transformer

        graph = name.removeprefix("gat_transformer").lstrip("_") or None  # None = config
        return build_gat_transformer(
            cfg,
            len(daily.modules),
            n_feat,
            n_cal,
            data["horizon"],
            max_lb,
            build_graph(cfg).edge_index(),
            graph,
        )
    if name in {"stgcn", "dcrnn"}:
        from src.graph.sap_graph import build_graph
        from src.models.stgnn import build_stgnn

        return build_stgnn(
            name,
            cfg,
            len(daily.modules),
            n_feat,
            n_cal,
            data["horizon"],
            build_graph(cfg).edge_index(),
        )
    raise ValueError(f"unknown model {name!r}")


def model_config(cfg: dict, name: str) -> dict:
    if name.startswith("gat_transformer"):
        m = dict(cfg["models"]["gat_transformer"])
        m["graph"] = name.removeprefix("gat_transformer").lstrip("_") or m["graph"]
        return m
    return cfg["models"][name]


def explain_artifacts(model, windows: dict, modules: list[str]) -> dict:
    """GAT attention per edge (averaged over test windows and heads) and the learned
    adjacency, saved with each run so Phase 6 can compare them with the true propagation."""
    import torch

    out = {}
    if hasattr(model, "learned_adjacency") and model.learned_adjacency() is not None:
        out["learned_adjacency_source_to_target"] = model.learned_adjacency().tolist()
    if hasattr(model, "gat"):
        model.eval()
        with torch.no_grad():
            model(
                torch.as_tensor(windows["X"]), torch.as_tensor(windows["Xf"]), keep_attention=True
            )
        att = model.last_attention
        src, dst = model.edge_index.tolist()
        out["gat_attention"] = [
            {"source": modules[s], "target": modules[t], "weight": float(w)}
            for s, t, w in zip(src, dst, att["edges"].mean(-1).tolist(), strict=True)
        ]
        out["gat_self_attention"] = dict(zip(modules, att["self"].mean(-1).tolist(), strict=True))
    return out


def run_name(model: str, target: str, seed: int, split: str = "main") -> str:
    return f"{model}_{target}_seed{seed}" + ("" if split == "main" else f"_{split}")


def variant_name(cfg: dict, name: str, lookback: int | None, drop_groups: tuple) -> str:
    """Display name of an ablation, e.g. gat_transformer__lb14 or gat_transformer__noaux."""
    out = name
    if lookback and lookback != cfg["dataset"]["lookback"]:
        out += f"__lb{lookback}"
    if "aux" in drop_groups:
        out += "__noaux"
    return out


def run_neural(
    name: str,
    cfg: dict,
    daily: DailyData,
    split: Split,
    target: str,
    seed: int,
    verbose: bool = True,
    time_budget: float | None = None,
    resume_dir: str | None = None,
    lookback: int | None = None,
    drop_groups: tuple[str, ...] = (),
) -> dict | None:
    """Train one (model, target, seed) and save its results. Returns None if training was
    paused by `time_budget`; rerunning the same command then resumes it."""
    import torch

    from src.models.trainer import fit, predict

    set_seed(seed)
    ev, horizons = cfg["evaluation"], cfg["dataset"]["horizons"]
    display = variant_name(cfg, name, lookback, drop_groups)
    rid = run_name(display, target, seed, split.name)
    data = prepare(cfg, daily, split, target, lookback=lookback, drop_groups=drop_groups)
    model = build_model(name, cfg, daily, data)
    resume = Path(resume_dir) if resume_dir else get_path(cfg, "checkpoints")
    resume.mkdir(parents=True, exist_ok=True)
    partial = resume / f"{rid}.partial.pt"
    stats = fit(model, data, target, cfg["training"], seed, verbose, partial, time_budget)
    if stats["status"] == "paused":
        print(f"PAUSED {rid} after epoch {stats['epochs_run']} - rerun the same command to resume")
        return None
    try:
        partial.unlink(missing_ok=True)
    except OSError:  # e.g. a folder where deleting is not allowed: harmless leftover
        pass
    stats.pop("status")
    objective = stats.pop("objective")
    history = stats.pop("history")
    parts = ["val", "test"]
    preds = {p: predict(model, data[p], objective) for p in parts}
    truth = {p: data[p]["Y"] for p in parts}
    record = {
        "model": display,
        "base_model": name,
        "split": split.name,
        "target": target,
        "seed": seed,
        "data": "SYNTHETIC DATA" if daily.synthetic else "REAL DATA",
        "metrics": score(
            target, truth, preds, horizons, ev["precision_at_k"], ev["deviance_floor"]
        ),
        "training": stats,
        "history": history,
        "config": {
            "model": model_config(cfg, name),
            "training": cfg["training"],
            "lookback": data["lookback"],
            "features": data["feature_names"],
        },
        "explain": explain_artifacts(model, data["test"], daily.modules),
    }
    runs = get_path(cfg, "results") / "runs"
    runs.mkdir(exist_ok=True)
    with open(runs / f"{rid}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    pred_dir = get_path(cfg, "results") / "predictions"
    pred_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        pred_dir / f"{rid}.npz",
        val=preds["val"],
        test=preds["test"],
        val_origins=split.val,
        test_origins=split.test,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": name,
            "target": target,
            "seed": seed,
            "scaler": data["scaler"].__dict__,
            "feature_names": data["feature_names"],
        },
        get_path(cfg, "checkpoints") / f"{rid}.pt",
    )
    m = record["metrics"]["test"]["h1"]
    key = "MAE" if target == "regression" else "PR_AUC"
    print(
        f"{rid}: best epoch {stats['best_epoch']}/{stats['epochs_run']}, "
        f"{stats['train_seconds']}s, {stats['n_params']} params, test {key}@1 = {m[key]:.3f}"
    )
    return record


# --------------------------------------------------------------------------- comparison
def summarise(cfg: dict, targets: list[str]) -> dict[str, pd.DataFrame]:
    """Test-split comparison: baselines (deterministic, 1 run) and neural models (mean ± std)."""
    horizons = cfg["dataset"]["horizons"]
    results_dir = get_path(cfg, "results")
    base = json.loads((results_dir / "baselines.json").read_text(encoding="utf-8"))
    runs = (
        [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((results_dir / "runs").glob("*.json"))
        ]
        if (results_dir / "runs").exists()
        else []
    )
    runs = [r for r in runs if r.get("split", "main") == "main"]  # folds: see evaluate.py
    tables, md = {}, [f"# Model comparison - test split ({base.get('data', '')})", ""]
    for target in targets:
        rows = []
        for model in BASELINES:
            m = base.get(target, {}).get("test", {}).get(model)
            if m:
                rows.append(
                    {
                        "model": model,
                        "n_seeds": 1,
                        **{
                            f"{k}@{h}": (m[f"h{h}"][k], np.nan)
                            for k in REPORTED[target]
                            for h in horizons
                        },
                    }
                )
        by_model: dict[str, list] = {}
        for r in runs:
            if r["target"] == target:
                by_model.setdefault(r["model"], []).append(r["metrics"]["test"])
        for model, ms in by_model.items():
            row = {"model": model, "n_seeds": len(ms)}
            for k in REPORTED[target]:
                for h in horizons:
                    v = np.array([m[f"h{h}"][k] for m in ms])
                    row[f"{k}@{h}"] = (
                        float(v.mean()),
                        float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                    )
            rows.append(row)
        if not rows:
            continue
        cols = [c for c in rows[0] if "@" in c]
        flat = pd.DataFrame(
            [
                {
                    "model": r["model"],
                    "n_seeds": r["n_seeds"],
                    **{f"{c}_mean": r[c][0] for c in cols},
                    **{f"{c}_std": r[c][1] for c in cols},
                }
                for r in rows
            ]
        )
        flat.to_csv(results_dir / f"comparison_{target}.csv", index=False)
        tables[target] = flat

        def cell(r, c, rows=rows):
            mean, std = r[c]
            vals = [x[c][0] for x in rows]
            best = min(vals) if c.split("@")[0] in LOWER_IS_BETTER else max(vals)
            s = f"{mean:.3f}" + (f" ± {std:.3f}" if not np.isnan(std) else "")
            return f"**{s}**" if np.isclose(mean, best) else s

        md += [
            f"## {target}",
            "",
            "| model | seeds | " + " | ".join(cols) + " |",
            "|" + "---|" * (len(cols) + 2),
        ]
        md += [
            f"| {r['model']} | {r['n_seeds']} | " + " | ".join(cell(r, c) for c in cols) + " |"
            for r in rows
        ]
        if base.get(f"lightgbm_{target}", {}).get("status", "ok") != "ok":
            md.append(f"\nLightGBM: {base[f'lightgbm_{target}']['status']}")
        md.append("")
    md += [
        "Best value per column in bold. ± = standard deviation over seeds.",
        "",
        *[f"- {n}" for n in NOTES],
    ]
    (results_dir / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    return tables


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models",
        default="all",
        help=f"comma list from: baselines, {', '.join(NEURAL + GAT_VARIANTS)}; or 'all'",
    )
    ap.add_argument("--target", choices=["regression", "classification", "both"], default="both")
    ap.add_argument("--seeds", default=None, help="comma list; default: project.seeds")
    ap.add_argument("--summary", action="store_true", help="only rebuild comparison tables")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="seconds; pause a neural run after the epoch that crosses it",
    )
    ap.add_argument(
        "--resume-dir",
        default=None,
        help="where partial checkpoints live (default: paths.checkpoints)",
    )
    ap.add_argument("--lookback", type=int, default=None, help="ablation: look-back in days")
    ap.add_argument("--no-aux", action="store_true", help="ablation: drop auxiliary signals")
    ap.add_argument(
        "--split",
        default="main",
        help="'main' or a rolling-origin fold: fold1 .. foldK (neural models)",
    )
    args = ap.parse_args(argv)

    cfg = load_config()
    targets = ["regression", "classification"] if args.target == "both" else [args.target]
    if not args.summary:
        try:
            daily = load_daily(cfg)
        except FileNotFoundError:
            print("No processed dataset found: run `python run.py data` first.")
            return 1
        ds = cfg["dataset"]
        lookback = args.lookback or ds["lookback"]
        if args.split == "main":
            split = chronological_split(
                len(daily.dates), lookback, max(ds["horizons"]), ds["split"]
            )
        else:
            ro = ds["rolling_origin"]
            folds = {
                f.name: f
                for f in rolling_origin_splits(
                    len(daily.dates),
                    lookback,
                    max(ds["horizons"]),
                    ro["n_folds"],
                    ro["val_days"],
                    ro["test_days"],
                )
            }
            if args.split not in folds:
                ap.error(f"--split must be main or one of {sorted(folds)}")
            split = folds[args.split]
        models = ["baselines", *NEURAL] if args.models == "all" else args.models.split(",")
        unknown = set(models) - {"baselines", *NEURAL, *GAT_VARIANTS}
        if unknown:
            ap.error(f"unknown models: {sorted(unknown)}")
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["project"]["seeds"]
        if "baselines" in models:
            run_baseline_stage(cfg, daily, split, targets, cfg["project"]["seed"])
        jobs = [
            (name, target, seed)
            for name in models
            if name in NEURAL or name in GAT_VARIANTS
            for target in targets
            for seed in seeds
        ]
        for name, target, seed in jobs:
            rec = run_neural(
                name,
                cfg,
                daily,
                split,
                target,
                seed,
                verbose=not args.quiet,
                time_budget=args.time_budget,
                resume_dir=args.resume_dir,
                lookback=args.lookback,
                drop_groups=("aux",) if args.no_aux else (),
            )
            if rec is None:  # paused by the time budget: stop here, resume on the next call
                return 3

    tables = summarise(cfg, targets)
    for target, t in tables.items():
        show = t[["model", "n_seeds"] + [c for c in t.columns if c.endswith("_mean")]]
        show.columns = [c.replace("_mean", "") for c in show.columns]
        print(
            f"\nTEST split - {target} (mean over seeds)\n"
            + show.to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )
    print("\nsaved results/comparison.md and results/comparison_*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
