"""SYNTHETIC SAP incident data generator.

Real SAP ticket data is usually confidential, so the project first uses data from a
generator whose mechanism is known. That lets us check later whether a model learned the
mechanism (e.g. compare GAT attention with the true propagation matrix).

Daily incident count of module i on day t is a sum of independent Poisson parts:

  base        lam = base_rate_i * calendar_i(t) * stress_i(t)
                    calendar: weekend / Monday factors, month-end and quarter-end close
                    stress:   hidden AR(1) "system health" state (mean 1)
  change      lam = uplift * sum_d kernel[d] * change_i(t-d)          d = 1..3
  propagation lam = sum_j W[j, i] * sum_d kernel[d] * y_j(t-d)        d = 1..3
  (change and propagation parts are also scaled by the weekend reporting factor)

W is the TRUE propagation matrix (config `synthetic.propagation`); it is saved to
ground_truth_propagation.json. Because the parts are independent Poisson draws, every
incident has a known cause (ground_truth_incident_causes.csv). This is a discrete-time,
multivariate Hawkes-style process. Each part uses its own random stream, so switching
propagation off leaves the base and change incidents exactly the same (used in tests).

Auxiliary signals: job failures and IDoc errors are noisy readings of the hidden stress
state plus the day's job/interface incidents, so they carry some information that the
incident counts alone do not.

    python -m src.data.generate_synthetic
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from src.data import text_templates as tt  # noqa: E402
from src.data.schema import (  # noqa: E402
    INCIDENT_COLUMNS,
    build_calendar,
    validate_calendar,
    validate_daily_signals,
    validate_incidents,
)
from src.graph.sap_graph import build_graph  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402

STREAMS = ["latent", "change", "base", "change_counts", "propagation", "tickets", "aux"]
SYNTHETIC_LABEL = "SYNTHETIC DATA"


@dataclass
class SyntheticData:
    incidents: pd.DataFrame  # schema columns only
    signals: pd.DataFrame  # date, module, job_failures, idoc_errors, change_events
    calendar: pd.DataFrame
    causes: pd.DataFrame  # incident_id, cause, source_module   (ground truth)
    counts: pd.DataFrame  # index = date, columns = modules
    ground_truth: dict
    rates: pd.DataFrame | None = None  # true expected counts (date, module, rate, ripple_rate)


# --------------------------------------------------------------------------- helpers
def per_module(spec, modules: list[str], default: float = 1.0) -> np.ndarray:
    """Config value -> array in module order. Accepts a number or {default: x, FI: y, ...}."""
    if not isinstance(spec, dict):
        return np.full(len(modules), float(spec))
    d = spec.get("default", default)
    return np.array([float(spec.get(m, d)) for m in modules])


def propagation_matrix(spec: list, modules: list[str]) -> np.ndarray:
    """[N, N] true propagation matrix, row = source, column = target."""
    idx = {m: i for i, m in enumerate(modules)}
    w = np.zeros((len(modules), len(modules)))
    for src, dst, strength in spec:
        if src not in idx or dst not in idx or src == dst:
            raise ValueError(f"bad propagation entry {src}->{dst}")
        w[idx[src], idx[dst]] = float(strength)
    return w


def _shift(x: np.ndarray, d: int) -> np.ndarray:
    """x shifted d days later (row t holds x[t-d]); first d rows are zero."""
    out = np.zeros_like(x, dtype=float)
    out[d:] = x[:-d]
    return out


# --------------------------------------------------------------------------- simulation
def simulate_counts(
    scfg: dict, modules: list[str], cal: pd.DataFrame, rngs: dict, propagation: bool = True
) -> dict[str, np.ndarray]:
    t_days, n = len(cal), len(modules)
    weekend = cal["is_weekend"].to_numpy()
    close = cal["is_month_end_close"].to_numpy()
    quarter = cal["is_quarter_end"].to_numpy()
    monday = (cal["date"].dt.dayofweek == 0).to_numpy()

    # Calendar multiplier per day and module
    close_f = per_module(scfg["close_factor"], modules, default=1.0)
    mult = np.ones((t_days, n))
    mult[weekend] = per_module(scfg["weekend_factor"], modules)
    # Fewer users work at weekends, so fewer problems of ANY cause get reported then.
    reporting = np.where(weekend[:, None], mult, 1.0)
    mult[monday] *= scfg["monday_factor"]
    mult[close] *= close_f
    mult[np.ix_(quarter, close_f > 1)] *= scfg["quarter_end_factor"]

    # Hidden stress state: AR(1), exponentiated and centred so its mean multiplier is 1
    phi, sigma = scfg["latent_stress"]["phi"], scfg["latent_stress"]["sigma"]
    eps = rngs["latent"].normal(0.0, sigma, (t_days, n))
    s = np.zeros((t_days, n))
    s[0] = eps[0] / np.sqrt(1 - phi**2)
    for t in range(1, t_days):
        s[t] = phi * s[t - 1] + eps[t]
    stress = np.exp(s - sigma**2 / (1 - phi**2) / 2)

    lam_base = per_module(scfg["base_rate"], modules) * mult * stress
    base_counts = rngs["base"].poisson(lam_base)

    # Change events (weekdays only) raise the module's rate on the next 1-3 days
    kernel = np.asarray(scfg["lag_kernel"], dtype=float)
    p_change = per_module(scfg["change"]["daily_prob"], modules, default=0.0)
    changes = (rngs["change"].random((t_days, n)) < p_change) & ~weekend[:, None]
    lam_change = (
        reporting
        * scfg["change"]["uplift"]
        * sum(k * _shift(changes, d) for d, k in enumerate(kernel, start=1))
    )
    change_counts = rngs["change_counts"].poisson(lam_change)

    # Propagation: yesterday's (and 2-3 days ago) incidents in j raise today's rate in i
    w = propagation_matrix(scfg["propagation"], modules) if propagation else np.zeros((n, n))
    prop = np.zeros((t_days, n, n), dtype=np.int64)  # [day, source, target]
    y = np.zeros((t_days, n), dtype=np.int64)
    lam_ripple = np.zeros((t_days, n))  # expected ripple incidents per target (ground truth)
    for t in range(t_days):
        lam = np.zeros((n, n))
        for d, k in enumerate(kernel, start=1):
            if t - d >= 0:
                lam += k * y[t - d][:, None] * w
        prop[t] = rngs["propagation"].poisson(lam * reporting[t])
        lam_ripple[t] = (lam * reporting[t]).sum(axis=0)
        y[t] = base_counts[t] + change_counts[t] + prop[t].sum(axis=0)

    return {
        "y": y, "base": base_counts, "change": change_counts, "prop": prop,
        "changes": changes.astype(np.int64), "stress": stress, "W": w,
        # true expected count of each module-day given everything up to that day: the best
        # possible forecast (an "oracle"), used only to measure how much is predictable
        "rate": lam_base + lam_change + lam_ripple, "ripple_rate": lam_ripple,
    }  # fmt: skip


# --------------------------------------------------------------------------- tickets
def make_incidents(
    sim: dict, cal: pd.DataFrame, modules: list[str], scfg: dict, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # One row per incident: (day, module, cause, source module index or -1)
    days, mods, causes, srcs = [], [], [], []
    for cause in ["base", "change"]:
        t, i = np.nonzero(sim[cause])
        reps = sim[cause][t, i]
        days.append(np.repeat(t, reps))
        mods.append(np.repeat(i, reps))
        causes.append(np.full(reps.sum(), cause, dtype=object))
        srcs.append(np.full(reps.sum(), -1))
    t, j, i = np.nonzero(sim["prop"])
    reps = sim["prop"][t, j, i]
    days.append(np.repeat(t, reps))
    mods.append(np.repeat(i, reps))
    causes.append(np.full(reps.sum(), "propagation", dtype=object))
    srcs.append(np.repeat(j, reps))
    day, mod, cause, src = (np.concatenate(a) for a in (days, mods, causes, srcs))
    m_total = len(day)

    # Time of day: mostly business hours on weekdays
    weekend = cal["is_weekend"].to_numpy()[day]
    business = rng.random(m_total) < np.where(weekend, 0.5, 0.85)
    hours = np.where(business, rng.uniform(8, 18, m_total), rng.uniform(0, 24, m_total))
    created = pd.to_datetime(cal["date"].to_numpy()[day]) + pd.to_timedelta(
        np.round(hours * 3600), unit="s"
    )

    # Priority: module-specific mix, more P1/P2 during close for the close modules
    probs = scfg["priority"]["probs"]
    close_mod = per_module(scfg["close_factor"], modules, default=1.0) > 1
    close_day = cal["is_month_end_close"].to_numpy()[day]
    boost = scfg["priority"]["close_p1p2_boost"]
    cats = tt.CATEGORIES
    mix = {m: np.array(tt.CATEGORY_MIX[m]) / sum(tt.CATEGORY_MIX[m]) for m in modules}

    priority, category, desc, notes = [], [], [], []
    for k in range(m_total):
        m = modules[mod[k]]
        p = np.array(probs.get(m, probs["default"]), dtype=float)
        if close_day[k] and close_mod[mod[k]]:
            p[:2] *= boost
        priority.append(f"P{rng.choice(4, p=p / p.sum()) + 1}")
        # Category depends on the cause: changes tend to cause transport regressions,
        # cross-module ripples tend to show up as interface or posting errors.
        if cause[k] == "change" and rng.random() < 0.6:
            c = "transport_change"
        elif cause[k] == "propagation" and rng.random() < 0.5:
            c = rng.choice(["interface_idoc", "posting_error"])
        else:
            c = cats[rng.choice(len(cats), p=mix[m])]
        category.append(str(c))
        src_name = modules[src[k]] if src[k] >= 0 else None
        d, r = tt.describe(m, str(c), rng, scfg["vague_description_share"], src_name)
        desc.append(d)
        notes.append(r)

    # Resolution time: log-normal around a priority-specific median; tickets that would be
    # resolved after the data extraction date are still open (resolved_at missing).
    med = scfg["resolution_hours_median"]
    res_hours = np.array([med[p] for p in priority]) * np.exp(rng.normal(0, 0.8, m_total))
    resolved = created + pd.to_timedelta(np.round(res_hours * 3600), unit="s")
    extraction = cal["date"].iloc[-1] + pd.Timedelta(days=1)
    resolved = resolved.where(resolved < extraction)

    df = pd.DataFrame({
        "created_at": created, "module": [modules[i] for i in mod], "priority": priority,
        "category": category, "short_description": desc, "resolution_notes": notes,
        "resolved_at": resolved, "_cause": cause,
        "_source": [modules[j] if j >= 0 else None for j in src],
    })  # fmt: skip
    df = df.sort_values(["created_at", "module"], kind="stable").reset_index(drop=True)
    df.insert(0, "incident_id", [f"INC{k + 1:07d}" for k in range(len(df))])
    causes_df = df[["incident_id", "_cause", "_source"]].rename(
        columns={"_cause": "cause", "_source": "source_module"}
    )
    return df[list(INCIDENT_COLUMNS)], causes_df


def make_signals(
    sim: dict, incidents: pd.DataFrame, cal: pd.DataFrame, modules: list[str], scfg: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:  # fmt: skip
    dates = cal["date"]

    def category_counts(category: str) -> np.ndarray:
        sub = incidents[incidents["category"] == category]
        grid = pd.crosstab(sub["created_at"].dt.normalize(), sub["module"])
        return grid.reindex(index=dates, columns=modules, fill_value=0).to_numpy()

    aux = scfg["aux"]
    k = aux["per_incident"]
    jf = rng.poisson(
        per_module(aux["job_failures_base"], modules) * sim["stress"]
        + k * category_counts("job_failure")
    )
    idoc = rng.poisson(
        per_module(aux["idoc_errors_base"], modules) * sim["stress"]
        + k * category_counts("interface_idoc")
    )
    t_days, n = jf.shape
    return pd.DataFrame({
        "date": np.repeat(dates.to_numpy(), n),
        "module": np.tile(modules, t_days),
        "job_failures": jf.ravel(),
        "idoc_errors": idoc.ravel(),
        "change_events": sim["changes"].ravel(),
    })  # fmt: skip


# --------------------------------------------------------------------------- main API
def generate(cfg: dict, seed: int | None = None, propagation: bool = True) -> SyntheticData:
    seed = cfg["project"]["seed"] if seed is None else seed
    scfg, modules = cfg["synthetic"], list(cfg["data"]["modules"])
    if list(cfg["data"]["categories"]) != tt.CATEGORIES:
        raise ValueError("config data.categories must match text_templates.CATEGORIES")

    rngs = dict(
        zip(
            STREAMS,
            (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(len(STREAMS))),
            strict=True,
        )
    )
    start = pd.Timestamp(scfg["start_date"])
    end = start + pd.Timedelta(days=scfg["n_days"] - 1)
    c = cfg["calendar"]
    cal = build_calendar(
        start,
        end,
        c["close_last_working_days"],
        c["close_first_working_days"],
        c["quarter_end_months"],
    )

    sim = simulate_counts(scfg, modules, cal, rngs, propagation)
    incidents, causes = make_incidents(sim, cal, modules, scfg, rngs["tickets"])
    signals = make_signals(sim, incidents, cal, modules, scfg, rngs["aux"])
    counts = pd.DataFrame(
        sim["y"], index=pd.DatetimeIndex(cal["date"], name="date"), columns=modules
    )
    rates = pd.DataFrame({
        "date": np.repeat(cal["date"].to_numpy(), len(modules)),
        "module": np.tile(modules, len(cal)),
        "rate": sim["rate"].ravel(), "ripple_rate": sim["ripple_rate"].ravel(),
    })  # fmt: skip
    return SyntheticData(incidents, signals, cal, causes, counts,
                         ground_truth(cfg, sim["W"], seed, propagation), rates)  # fmt: skip


def ground_truth(cfg: dict, w: np.ndarray, seed: int, propagation: bool) -> dict:
    modules = list(cfg["data"]["modules"])
    graph = set(build_graph(cfg).edges)
    true_edges = {(modules[i], modules[j]) for i, j in zip(*np.nonzero(w), strict=True)}
    listed = {(s, t) for s, t, _ in cfg["synthetic"]["propagation"]} | graph
    edges = [
        {"source": s, "target": t, "strength": float(w[modules.index(s), modules.index(t)]),
         "in_hand_built_graph": (s, t) in graph}
        for s, t in sorted(listed)
    ]  # fmt: skip
    return {
        "synthetic": True,
        "note": f"{SYNTHETIC_LABEL}: true mechanism of src/data/generate_synthetic.py",
        "seed": seed,
        "propagation_enabled": propagation,
        "modules": modules,
        "lag_kernel": cfg["synthetic"]["lag_kernel"],
        "matrix_source_to_target": w.round(6).tolist(),
        "edges": edges,
        "hand_built_edges_without_true_effect": sorted(map(list, graph - true_edges)),
        "true_edges_missing_from_hand_built_graph": sorted(map(list, true_edges - graph)),
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(w)))),
        "change_uplift": cfg["synthetic"]["change"]["uplift"],
    }


def validate(data: SyntheticData, cfg: dict) -> list:
    modules = cfg["data"]["modules"]
    return [
        validate_incidents(data.incidents, modules, cfg["data"]["categories"],
                           cfg["data"]["priorities"]),
        validate_daily_signals(data.signals, modules),
        validate_calendar(data.calendar),
    ]  # fmt: skip


def save(data: SyntheticData, out_dir) -> None:
    data.incidents.to_csv(out_dir / "incidents.csv", index=False)
    data.signals.to_csv(out_dir / "daily_signals.csv", index=False)
    data.calendar.to_csv(out_dir / "calendar.csv", index=False)
    data.causes.to_csv(out_dir / "ground_truth_incident_causes.csv", index=False)
    if data.rates is not None:
        data.rates.to_csv(out_dir / "ground_truth_rates.csv", index=False, float_format="%.6g")
    with open(out_dir / "ground_truth_propagation.json", "w", encoding="utf-8") as f:
        json.dump(data.ground_truth, f, indent=2)
    info = {
        "synthetic": True,
        "label": SYNTHETIC_LABEL,
        "generator": "src/data/generate_synthetic.py",
        "seed": data.ground_truth["seed"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "date_range": [str(data.calendar["date"].min().date()),
                       str(data.calendar["date"].max().date())],
        "n_incidents": len(data.incidents),
    }  # fmt: skip
    with open(out_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def load_synthetic(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the saved synthetic tables back with the schema's types."""
    d = get_path(cfg, "synthetic")
    incidents = pd.read_csv(d / "incidents.csv", parse_dates=["created_at", "resolved_at"])
    signals = pd.read_csv(d / "daily_signals.csv", parse_dates=["date"])
    calendar = pd.read_csv(d / "calendar.csv", parse_dates=["date"])
    return incidents, signals, calendar


# --------------------------------------------------------------------------- summary
def summarise(data: SyntheticData, cfg: dict) -> dict:
    counts, cal, inc = data.counts, data.calendar, data.incidents
    modules = list(counts.columns)
    weekday = ~cal["is_weekend"].to_numpy()
    close = cal["is_month_end_close"].to_numpy()
    p1p2 = inc["priority"].isin(["P1", "P2"])
    p1p2_days = (pd.crosstab(inc.loc[p1p2, "created_at"].dt.normalize(),
                             inc.loc[p1p2, "module"])
                 .reindex(index=counts.index, columns=modules, fill_value=0) > 0)  # fmt: skip
    sig = data.signals.set_index(["date", "module"])
    long_counts = counts.reset_index().melt(
        id_vars="date", var_name="module", value_name="incidents"
    )
    joined = sig.join(long_counts.set_index(["date", "module"]))

    def r(x):
        return round(float(x), 4)

    return {
        "synthetic": True,
        "date_range": [str(cal["date"].min().date()), str(cal["date"].max().date())],
        "n_days": len(cal),
        "n_incidents": len(inc),
        "incidents_per_day": r(counts.sum(axis=1).mean()),
        "share_p1_p2": r(p1p2.mean()),
        "share_open_at_extraction": r(inc["resolved_at"].isna().mean()),
        "causes_share": {k: r(v) for k, v in data.causes["cause"].value_counts(normalize=True)
                         .sort_index().items()},
        "per_module": {
            m: {
                "total": int(counts[m].sum()),
                "mean_per_day": r(counts[m].mean()),
                "share_zero_days": r((counts[m] == 0).mean()),
                "share_days_with_p1_p2": r(p1p2_days[m].mean()),
                "weekend_to_weekday_ratio": r(counts[m][~weekday].mean()
                                              / counts[m][weekday].mean()),
                "close_to_other_weekday_ratio": r(counts[m][weekday & close].mean()
                                                  / counts[m][weekday & ~close].mean()),
            }
            for m in modules
        },
        "aux_correlation_with_same_day_incidents": {
            c: r(joined[c].corr(joined["incidents"])) for c in ["job_failures", "idoc_errors"]
        },
        "aux_correlation_with_next_day_incidents": {
            c: r(joined[c].corr(joined.groupby(level="module")["incidents"].shift(-1)))
            for c in ["job_failures", "idoc_errors"]
        },
        "true_propagation_spectral_radius": r(data.ground_truth["spectral_radius"]),
    }  # fmt: skip


# --------------------------------------------------------------------------- plots
def plot_incidents_over_time(data: SyntheticData, path) -> None:
    counts = data.counts
    fig, axes = plt.subplots(5, 2, figsize=(13, 11), sharex=True)
    for ax, m in zip(axes.ravel(), counts.columns, strict=True):
        ax.plot(counts.index, counts[m], color="#c9d6e3", lw=0.6, label="daily")
        ax.plot(
            counts.index,
            counts[m].rolling(28, min_periods=7).mean(),
            color="#1f4e79",
            lw=1.4,
            label="28-day mean",
        )
        ax.set_title(m, loc="left", fontsize=10, fontweight="bold")
        ax.set_ylabel("incidents / day", fontsize=8)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    for ax in axes[-1]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.suptitle(f"{SYNTHETIC_LABEL} - incidents per day per SAP module", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_patterns(data: SyntheticData, path, close_modules: set[str]) -> None:
    counts, cal = data.counts, data.calendar
    weekday = ~cal["is_weekend"].to_numpy()
    close = cal["is_month_end_close"].to_numpy()
    by_dow = counts.groupby(counts.index.dayofweek).mean()
    by_dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    rel = by_dow / by_dow.loc[["Tue", "Wed", "Thu"]].mean()  # relative to mid-week

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    im = a1.imshow(rel.T.to_numpy(), cmap="Blues", aspect="auto", vmin=0)
    a1.set_xticks(range(7), rel.index)
    a1.set_yticks(range(len(rel.columns)), rel.columns)
    for (i, j), v in np.ndenumerate(rel.T.to_numpy()):
        a1.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if v > 0.8 else "black",
        )
    fig.colorbar(im, ax=a1, fraction=0.04)
    a1.set_title("Weekly pattern: mean incidents per weekday\n(relative to Tue-Thu mean)")

    ratio = counts[weekday & close].mean() / counts[weekday & ~close].mean()
    colors = ["#1f4e79" if m in close_modules else "#9db4cc" for m in counts.columns]
    a2.bar(counts.columns, ratio, color=colors)
    a2.axhline(1, color="black", lw=0.8, ls="--")
    a2.set_ylabel("close-day mean / other weekday mean")
    a2.set_title(
        "Month-end close effect (weekdays only)\n"
        "dark = modules with a close factor in the generator"
    )
    fig.suptitle(f"{SYNTHETIC_LABEL} - weekly and month-end patterns", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_propagation_example(data: SyntheticData, path, source: str = "SD") -> None:
    gt = data.ground_truth
    modules = gt["modules"]
    w = np.array(gt["matrix_source_to_target"])
    in_graph = {(e["source"], e["target"]) for e in gt["edges"] if e["in_hand_built_graph"]}

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 2, width_ratios=[1, 1.3], figure=fig)
    a0 = fig.add_subplot(gs[:, 0])
    im = a0.imshow(w, cmap="Oranges", vmin=0)
    a0.set_xticks(range(len(modules)), modules, rotation=45)
    a0.set_yticks(range(len(modules)), modules)
    a0.set_xlabel("target module")
    a0.set_ylabel("source module")
    for i, s in enumerate(modules):
        for j, t in enumerate(modules):
            if (s, t) in in_graph:
                a0.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#1f4e79", lw=1.2)
                )
            elif w[i, j] > 0:
                a0.text(
                    j, i, "!", ha="center", va="center", fontweight="bold", color="red", fontsize=16
                )
    fig.colorbar(im, ax=a0, fraction=0.04, label="true strength")
    a0.set_title(
        "True propagation matrix W\nblue box = edge in hand-built graph, "
        "! = real effect missing from it",
        fontsize=9,
    )

    # Cascade around the source module's busiest day (ground-truth causes)
    counts = data.counts
    peak = counts[source].iloc[3:-6].idxmax()
    window = pd.date_range(peak - pd.Timedelta(days=3), peak + pd.Timedelta(days=5))
    caused = data.incidents.merge(data.causes, on="incident_id")
    caused = caused[caused["source_module"] == source]
    caused = (pd.crosstab(caused["created_at"].dt.normalize(), caused["module"])
              .reindex(index=window, fill_value=0))  # fmt: skip
    labels = [d.strftime("%a %d %b") for d in window]

    a1 = fig.add_subplot(gs[0, 1])
    a1.bar(labels, counts.loc[window, source], color="#7f7f7f")
    a1.set_title(f"{source} incidents per day around its busiest day ({peak.date()})", fontsize=9)
    a1.tick_params(axis="x", labelbottom=False)
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    bottom = np.zeros(len(window))
    for col in caused.columns:
        a2.bar(labels, caused[col], bottom=bottom, label=col)
        bottom += caused[col].to_numpy()
    a2.set_title(
        f"incidents in other modules directly triggered by {source} (ground-truth causes)",
        fontsize=9,
    )
    a2.legend(fontsize=8, ncol=4)
    a2.tick_params(axis="x", rotation=45, labelsize=8)
    fig.suptitle(f"{SYNTHETIC_LABEL} - how incidents propagate between modules", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    seed = cfg["project"]["seed"]
    set_seed(seed)
    data = generate(cfg, seed)
    reports = validate(data, cfg)
    for rep in reports:
        print(rep)
    for rep in reports:
        rep.raise_if_errors()

    out = get_path(cfg, "synthetic")
    save(data, out)
    figs = get_path(cfg, "figures")
    plot_incidents_over_time(data, figs / "incidents_over_time.png")
    plot_patterns(
        data, figs / "weekly_monthend_patterns.png", set(cfg["synthetic"]["close_factor"])
    )
    plot_propagation_example(data, figs / "propagation_example.png")

    summary = summarise(data, cfg)
    with open(get_path(cfg, "results") / "data_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n{SYNTHETIC_LABEL}: {summary['n_incidents']} incidents over {summary['n_days']} "
        f"days ({summary['date_range'][0]} .. {summary['date_range'][1]}), "
        f"{summary['incidents_per_day']} per day"
    )
    print("causes:", summary["causes_share"])
    print(
        "missing from hand-built graph:",
        data.ground_truth["true_edges_missing_from_hand_built_graph"],
    )
    print(
        "in hand-built graph but no effect:",
        data.ground_truth["hand_built_edges_without_true_effect"],
    )
    print(f"saved data to {out}, figures to {figs}, summary to results/data_summary.json")


if __name__ == "__main__":
    main()
