"""Phase 6: evaluation, ablations, rolling-origin stability, explainability.

    python -m src.evaluate        (after `python -m src.train` has saved the runs)

Reads the saved runs and recomputes what needs no training (baselines on the rolling-origin
folds, the oracle ceiling). Writes results/summary.md, results/evaluation.json and figures.

Sections
  1 main results       every model x horizon, mean ± std over seeds (test split)
  2 ablations          main model with/without graph, hand-built vs learned adjacency,
                       look-back 14 vs 28, with/without auxiliary signals - as paired
                       differences against the main model with the same seed
  3 rolling origin     test MAE@1 on 4 forward-moving test windows
  4 oracle ceiling     (synthetic data only) forecasts from the generator's TRUE rates: the
                       best any model could do, and how much of that comes from ripples
  5 explainability     learned adjacency and GAT attention vs the true propagation matrix
  6 LightGBM           gain importance
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.data.build_dataset import (  # noqa: E402
    chronological_split,
    load_daily,
    rolling_origin_splits,
)
from src.metrics import regression_metrics  # noqa: E402
from src.models.baselines import run_baselines, target_windows  # noqa: E402
from src.train import summarise, to_markdown  # noqa: E402
from src.utils import get_path, load_config  # noqa: E402

MAIN = "gat_transformer"
ABLATIONS = {
    "gat_transformer": "main model: hand-built graph + learned adjacency, 28 days, aux on",
    "gat_transformer_hand_built": "hand-built graph only",
    "gat_transformer_adaptive": "learned adjacency only",
    "gat_transformer_none": "no graph (same network)",
    "gat_transformer__lb14": "look-back 14 days",
    "gat_transformer__noaux": "without job failures / IDoc errors / change events",
}
MAIN_ORDER = [
    "naive",
    "seasonal_naive",
    "historical_mean",
    "lightgbm",
    "lstm",
    "transformer",
    "dcrnn",
    "stgcn",
    "gat_transformer",
]


def load_runs(cfg: dict) -> list[dict]:
    d = get_path(cfg, "results") / "runs"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))]


def fmt(mean: float, std: float | None = None) -> str:
    return f"{mean:.3f}" + ("" if std is None or np.isnan(std) else f" ± {std:.3f}")


# --------------------------------------------------------------------------- 1 main table
def main_table(cfg: dict, tables: dict) -> dict[str, pd.DataFrame]:
    cols = {
        "regression": ["MAE@1", "MAE@3", "MAE@7", "PoissonDev@1", "PoissonDev@7", "P@3@1"],
        "classification": ["PR_AUC@1", "PR_AUC@3", "PR_AUC@7", "ROC_AUC@1", "P@3@1", "R@3@1"],
    }
    out = {}
    for target, t in tables.items():
        t = t.set_index("model")
        rows = [
            {
                "model": m,
                "seeds": int(t.loc[m, "n_seeds"]),
                **{c: fmt(t.loc[m, f"{c}_mean"], t.loc[m, f"{c}_std"]) for c in cols[target]},
            }
            for m in MAIN_ORDER
            if m in t.index
        ]
        out[target] = pd.DataFrame(rows)
    return out


def horizon_plot(tables: dict, path) -> None:
    t = tables["regression"].set_index("model")
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in ["historical_mean", "lightgbm", "lstm", "transformer", "dcrnn", "stgcn", MAIN]:
        if m in t.index:
            ax.plot(
                [1, 3, 7],
                [t.loc[m, f"MAE@{h}_mean"] for h in [1, 3, 7]],
                marker="o",
                label=m,
                lw=2.2 if m == MAIN else 1.2,
            )
    ax.set_xticks([1, 3, 7])
    ax.set_xlabel("forecast day t+h")
    ax.set_ylabel("test MAE (incidents per module-day)")
    ax.set_title("SYNTHETIC DATA - test MAE by horizon (mean over seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- 2 ablations
def ablations(runs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Absolute scores and paired differences (variant - main model, same seed and target)."""
    key = {"regression": ["MAE", "PoissonDev", "P@3"], "classification": ["PR_AUC", "R@3"]}
    by = {
        (r["model"], r["target"], r["seed"]): r["metrics"]["test"]["h1"]
        for r in runs
        if r.get("split", "main") == "main"
    }
    rows, diffs = [], []
    for name, desc in ABLATIONS.items():
        row = {"variant": name, "what": desc}
        drow = {"variant": name}
        for target, metrics in key.items():
            seeds = sorted(s for (m, t, s) in by if m == name and t == target)
            for k in metrics:
                col = f"{k}@1 ({'counts' if target == 'regression' else 'P1/P2'})"
                vals = [by[(name, target, s)][k] for s in seeds]
                row[col] = (
                    fmt(np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else None)
                    if vals
                    else "not run"
                )
                paired = [
                    by[(name, target, s)][k] - by[(MAIN, target, s)][k]
                    for s in seeds
                    if (MAIN, target, s) in by
                ]
                drow[col] = (
                    fmt(np.mean(paired), np.std(paired, ddof=1) if len(paired) > 1 else None)
                    if paired and name != MAIN
                    else "-"
                )
        rows.append(row)
        diffs.append(drow)
    return pd.DataFrame(rows), pd.DataFrame(diffs)


# --------------------------------------------------------------------------- 3 rolling origin
def rolling_origin(cfg: dict, daily, runs: list[dict], path) -> pd.DataFrame:
    ds, ro = cfg["dataset"], cfg["dataset"]["rolling_origin"]
    folds = rolling_origin_splits(
        len(daily.dates),
        ds["lookback"],
        max(ds["horizons"]),
        ro["n_folds"],
        ro["val_days"],
        ro["test_days"],
    )
    table: dict[str, dict] = {}
    for f in folds:
        res = run_baselines(cfg, daily, f, "regression", cfg["project"]["seed"])
        for m in ["historical_mean", "lightgbm"]:
            if m in res["metrics"]["test"]:
                table.setdefault(m, {})[f.name] = res["metrics"]["test"][m]["h1"]["MAE"]
    for r in runs:
        if r.get("split", "main").startswith("fold") and r["target"] == "regression":
            table.setdefault(r["model"], {})[r["split"]] = r["metrics"]["test"]["h1"]["MAE"]
    names = [f.name for f in folds]
    labels = {f.name: f"{f.name} ({daily.dates[f.val_end].date()}..)" for f in folds}
    rows = []
    for m, vals in table.items():
        v = [vals.get(n, np.nan) for n in names]
        rows.append(
            {
                "model": m,
                **{labels[n]: x for n, x in zip(names, v, strict=True)},
                "mean ± std": fmt(np.nanmean(v), np.nanstd(v, ddof=1)),
            }
        )
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, vals in table.items():
        ax.plot(range(1, len(names) + 1), [vals.get(n, np.nan) for n in names], marker="o", label=m)
    ax.set_xticks(range(1, len(names) + 1), [labels[n] for n in names], fontsize=7)
    ax.set_ylabel("test MAE@1")
    ax.set_title("SYNTHETIC DATA - rolling-origin test windows (60 days each)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return df


# --------------------------------------------------------------------------- 4 oracle
def oracle(cfg: dict, daily, split) -> dict | None:
    """Forecasts from the generator's true expected counts (synthetic data only).

    full oracle        knows each module-day's true rate (incl. the hidden stress state):
                       no real model can do better on average - an upper bound
    ripple-blind       same, but the ripple part is replaced by its training-period average
                       per (module, weekday): what is lost by knowing nothing about ripples
    """
    path = get_path(cfg, "synthetic") / "ground_truth_rates.csv"
    if not daily.synthetic or not path.exists():
        return None
    r = pd.read_csv(path, parse_dates=["date"])
    rate = r.pivot(index="date", columns="module", values="rate")[daily.modules].to_numpy()
    ripple = r.pivot(index="date", columns="module", values="ripple_rate")[daily.modules]
    dow = ripple.index.dayofweek
    avg = ripple.iloc[: split.train_end].groupby(dow[: split.train_end]).mean()
    blind = rate - ripple.to_numpy() + avg.loc[dow].to_numpy()
    horizons, ev = cfg["dataset"]["horizons"], cfg["evaluation"]
    truth = target_windows(daily.counts.astype(float), split.test, max(horizons))
    out = {}
    for name, series in [("oracle_full", rate), ("oracle_ripple_blind", blind)]:
        pred = target_windows(series, split.test, max(horizons))
        out[name] = regression_metrics(
            truth, pred, horizons, ev["precision_at_k"], ev["deviance_floor"]
        )
    test = slice(split.val_end, split.test_end)
    y, lam = daily.counts[test].astype(float), rate[test]
    out["share_of_variance_predictable"] = float(np.var(lam) / np.var(y))
    out["ripple_share_of_expected_incidents"] = float(ripple.to_numpy()[test].sum() / lam.sum())
    return out


# --------------------------------------------------------------------------- 5 explainability
def per_target_spearman(scores: np.ndarray, w: np.ndarray, sources_of: dict) -> float:
    """Mean over targets (with >= 3 candidate sources) of the rank correlation between the
    model's weights and the true strengths of that target's sources. Comparing within a
    target removes the effect of per-target normalisation (weights into a target sum to 1)."""
    vals = []
    for j, srcs in sources_of.items():
        if len(srcs) >= 3 and np.ptp(w[srcs, j]) > 0:
            vals.append(spearmanr(scores[srcs, j], w[srcs, j]).statistic)
    return float(np.nanmean(vals)) if vals else float("nan")


def explainability(cfg: dict, runs: list[dict], path) -> dict | None:
    gt_path = get_path(cfg, "synthetic") / "ground_truth_propagation.json"
    if not gt_path.exists():
        return None
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    mods = gt["modules"]
    w = np.array(gt["matrix_source_to_target"])
    n = len(mods)
    off = ~np.eye(n, dtype=bool)
    true_edge = (w > 0) & off
    out = {}
    learned = {}
    for variant in ["gat_transformer", "gat_transformer_adaptive"]:
        mats = [
            np.array(r["explain"]["learned_adjacency_source_to_target"])
            for r in runs
            if r["model"] == variant
            and r.get("split", "main") == "main"
            and "learned_adjacency_source_to_target" in r.get("explain", {})
        ]
        if mats:
            a = np.mean(mats, axis=0)
            learned[variant] = a
            rank_of = {
                f"{s}->{t}": int((a[off] > a[mods.index(s), mods.index(t)]).sum()) + 1
                for s, t in gt["true_edges_missing_from_hand_built_graph"]
            }
            out[f"learned_adjacency_{variant}"] = {
                "runs": len(mats),
                "spearman_with_true_W": float(spearmanr(a[off], w[off]).statistic),
                "auc_true_edges_vs_non_edges": float(roc_auc_score(true_edge[off], a[off])),
                "rank_of_hidden_edges_out_of_90": rank_of,
                "per_target_spearman_all_sources": per_target_spearman(
                    a, w, {j: [i for i in range(n) if i != j] for j in range(n)}
                ),
            }
    att_runs = [
        r
        for r in runs
        if r["model"] in {"gat_transformer", "gat_transformer_hand_built"}
        and r.get("split", "main") == "main"
        and r.get("explain", {}).get("gat_attention")
    ]
    att = None
    if att_runs:
        att = np.zeros((n, n))
        for r in att_runs:
            for e in r["explain"]["gat_attention"]:
                att[mods.index(e["source"]), mods.index(e["target"])] += e["weight"] / len(att_runs)
        graph_edges = [
            (mods.index(e["source"]), mods.index(e["target"]))
            for e in gt["edges"]
            if e["in_hand_built_graph"]
        ]
        a_vals = np.array([att[i, j] for i, j in graph_edges])
        in_edges = {j: [i for i, jj in graph_edges if jj == j] for j in range(n)}
        w_vals = np.array([w[i, j] for i, j in graph_edges])
        qm = (mods.index("QM"), mods.index("MM"))
        out["gat_attention"] = {
            "runs": len(att_runs),
            "spearman_with_true_strength_on_graph_edges": float(
                spearmanr(a_vals, w_vals).statistic
            ),
            "per_target_spearman_on_graph_edges": per_target_spearman(att, w, in_edges),
            "QM->MM_attention (true strength 0)": float(att[qm]),
            "mean_attention_other_edges_into_MM": float(
                np.mean([att[i, j] for i, j in graph_edges if j == qm[1] and i != qm[0]])
            ),
        }
    panels = [("true propagation W", w)] + [
        (f"learned adjacency\n{k}", v) for k, v in learned.items()
    ]
    if att is not None:
        panels.append(("GAT attention\n(hand-built edges)", att))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    for ax, (title, m) in zip(np.atleast_1d(axes), panels, strict=True):
        shown = np.where(off, m, np.nan)  # diagonal (self) hidden for comparability
        ax.imshow(shown, cmap="Oranges")
        ax.set_xticks(range(n), mods, rotation=90, fontsize=7)
        ax.set_yticks(range(n), mods, fontsize=7)
        ax.set_xlabel("target", fontsize=8)
        ax.set_ylabel("source", fontsize=8)
        ax.set_title(title, fontsize=9)
    fig.suptitle("SYNTHETIC DATA - what the models learned vs the true ripple matrix", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- 6 LightGBM
def lightgbm_importance(cfg: dict, path) -> pd.DataFrame | None:
    rd = get_path(cfg, "results")
    frames = []
    for target in ["regression", "classification"]:
        f = rd / f"lightgbm_importance_{target}.csv"
        if f.exists():
            d = pd.read_csv(f)
            d = d[d["horizon"] == 1]
            frames.append(
                pd.DataFrame(
                    {"feature": d["feature"], target: d["gain"] / d["gain"].sum()}
                ).set_index("feature")
            )
    if not frames:
        return None
    df = pd.concat(frames, axis=1).fillna(0).sort_values(frames[0].columns[0], ascending=False)
    top = df.head(12)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    top[::-1].plot.barh(ax=ax)
    ax.set_xlabel("share of total gain (h = 1)")
    ax.set_title("SYNTHETIC DATA - LightGBM feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return top.round(3).reset_index()


# --------------------------------------------------------------------------- main
def main() -> None:
    cfg = load_config()
    daily = load_daily(cfg)
    ds = cfg["dataset"]
    split = chronological_split(len(daily.dates), ds["lookback"], max(ds["horizons"]), ds["split"])
    label = "SYNTHETIC DATA" if daily.synthetic else "REAL DATA"
    figs, rd = get_path(cfg, "figures"), get_path(cfg, "results")
    runs = load_runs(cfg)
    tables = summarise(cfg, ["regression", "classification"])

    main_t = main_table(cfg, tables)
    horizon_plot(tables, figs / "horizon_mae.png")
    abl, abl_diff = ablations(runs)
    ro = rolling_origin(cfg, daily, runs, figs / "rolling_origin.png")
    orc = oracle(cfg, daily, split)
    exp = explainability(cfg, runs, figs / "explain_adjacency.png")
    imp = lightgbm_importance(cfg, figs / "lightgbm_importance.png")

    evaluation = {
        "data": label,
        "oracle": orc,
        "explainability": exp,
        "rolling_origin": ro.to_dict(orient="records"),
        "ablations": abl.to_dict(orient="records"),
        "ablation_paired_differences": abl_diff.to_dict(orient="records"),
    }
    (rd / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    md = [
        f"# Results summary ({label})",
        "",
        f"> **{label}.** "
        + (
            "All numbers come from synthetic data produced by "
            "`src/data/generate_synthetic.py`; they say nothing about a real SAP system."
            if daily.synthetic
            else "Real incident data."
        ),
        "",
        "Generated by `python -m src.evaluate` from the files in `results/`. Test split "
        f"{split.describe(daily.dates)['periods']['test'][0]} .. "
        f"{split.describe(daily.dates)['periods']['test'][1]}; ± = std over seeds.",
        "",
        "## 1. Main results (test split)",
        "",
    ]
    for target, t in main_t.items():
        md += [f"### {target}", "", to_markdown(t), ""]
    md += [
        "![MAE by horizon](figures/horizon_mae.png)",
        "",
        "## 2. Ablations of the main model (GAT-Transformer)",
        "",
        to_markdown(abl),
        "",
        "Paired difference to the main model (same seed; MAE / deviance: negative = "
        "better; PR-AUC / P@3 / R@3: positive = better):",
        "",
        to_markdown(abl_diff),
        "",
        "## 3. Rolling-origin stability (test MAE@1, counts)",
        "",
        to_markdown(ro, digits=3),
        "",
        "![rolling origin](figures/rolling_origin.png)",
        "",
    ]
    if orc:
        o = pd.DataFrame(
            [
                {
                    "forecast": k,
                    **{
                        f"{m}@{h}": orc[k][f"h{h}"][m]
                        for m in ["MAE", "PoissonDev", "P@3"]
                        for h in [1, 7]
                    },
                }
                for k in ["oracle_full", "oracle_ripple_blind"]
            ]
        )
        md += [
            "## 4. Oracle ceiling (synthetic only)",
            "",
            "Forecasts made from the generator's true expected counts. The full oracle also "
            "knows the hidden stress state, so no model can beat it on average.",
            "",
            to_markdown(o),
            "",
            f"- Share of test-period variance in daily counts that is predictable even in "
            f"principle (variance of the true rate / variance of the counts): "
            f"{orc['share_of_variance_predictable']:.3f}",
            f"- Ripples' share of expected test incidents: "
            f"{orc['ripple_share_of_expected_incidents']:.3f}",
            "",
        ]
    if exp:
        md += [
            "## 5. Explainability vs the true propagation matrix",
            "",
            "```json",
            json.dumps(exp, indent=2),
            "```",
            "",
            "![explainability](figures/explain_adjacency.png)",
            "",
        ]
    if imp is not None:
        md += [
            "## 6. LightGBM feature importance (gain share, h = 1)",
            "",
            to_markdown(imp),
            "",
            "![importance](figures/lightgbm_importance.png)",
            "",
        ]
    (rd / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:6]))
    if orc:
        print(
            "oracle MAE@1 full / ripple-blind:",
            round(orc["oracle_full"]["h1"]["MAE"], 3),
            round(orc["oracle_ripple_blind"]["h1"]["MAE"], 3),
            "predictable variance share:",
            round(orc["share_of_variance_predictable"], 3),
        )
    if exp:
        print(json.dumps(exp, indent=1))
    print("saved results/summary.md, results/evaluation.json, results/figures/*")


if __name__ == "__main__":
    main()
