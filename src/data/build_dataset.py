"""Model-ready dataset: daily tensor, features, targets, time splits, windows.

Shapes (T days, N modules, F features, L look-back, H = max horizon, C calendar features):
  features        [T, N, F]   one row per day and module, using data up to that day only
  counts / p1p2   [T, N]      regression / classification target sources
  future_calendar [T, C]      calendar of each day; known in advance, so it may be used for
                              the target days of a forecast (that is not leakage)
  windows         X [S, L, N, F]   days t-L+1 .. t       (t = forecast origin)
                  Y [S, H, N]      days t+1 .. t+H       (counts, or 1 if >= 1 P1/P2)
                  Xf [S, H, C]     calendar of days t+1 .. t+H

Leakage rules implemented here:
  * every feature on day d uses data from days <= d only (rolling means are backward-looking);
  * a sample belongs to a split only if ALL of its target days lie inside that split;
  * the scaler (log1p + z-score) is fit on training-period days only.

    python -m src.data.build_dataset
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.load_real import load_real, load_real_signals  # noqa: E402
from src.data.schema import PRIORITIES, build_calendar  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402

CALENDAR_FEATURES = ["is_weekend", "is_month_end_close", "is_quarter_end", "dow_sin", "dow_cos"]
AUX_FEATURES = ["job_failures", "idoc_errors", "change_events"]
SCALED_GROUPS = ["counts", "aux", "rolling"]  # count-like groups: log1p + z-score


@dataclass
class DailyData:
    dates: pd.DatetimeIndex
    modules: list[str]
    feature_names: list[str]
    groups: dict[str, list[str]]  # feature group -> names (for ablations)
    features: np.ndarray  # [T, N, F] float32, unscaled
    counts: np.ndarray  # [T, N] all incidents
    p1p2: np.ndarray  # [T, N] P1 + P2 incidents
    future_calendar: np.ndarray  # [T, C]
    synthetic: bool

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.features.shape

    def target(self, kind: str) -> np.ndarray:
        """[T, N] series the windows take their targets from."""
        if kind == "regression":
            return self.counts.astype(np.float32)
        if kind == "classification":
            return (self.p1p2 > 0).astype(np.float32)
        raise ValueError(f"unknown target {kind!r}")

    def feature_index(self, names: list[str]) -> np.ndarray:
        return np.array([self.feature_names.index(n) for n in names], dtype=int)


# --------------------------------------------------------------------------- loading
def load_source(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, bool]:
    """(incidents, daily signals or None, calendar, is_synthetic) for data.source."""
    source = cfg["data"]["source"]
    if source == "synthetic":
        from src.data.generate_synthetic import load_synthetic

        incidents, signals, calendar = load_synthetic(cfg)
        return incidents, signals, calendar, True
    if source == "real":
        incidents = load_real(cfg)
        if incidents is None:  # never fall back silently: results would be mislabelled
            raise FileNotFoundError("data.source is 'real' but no export was found")
        signals = load_real_signals(cfg)
        c = cfg["calendar"]
        calendar = build_calendar(
            incidents["created_at"].min(),
            incidents["created_at"].max(),
            c["close_last_working_days"],
            c["close_first_working_days"],
            c["quarter_end_months"],
        )
        return incidents, signals, calendar, False
    raise ValueError(f"data.source must be 'synthetic' or 'real', not {source!r}")


# --------------------------------------------------------------------------- daily tensor
def _grid(df: pd.DataFrame, dates: pd.DatetimeIndex, modules: list[str], cols: list[str]):
    """Long (date, module) table -> [T, N, len(cols)] array, missing cells = 0."""
    full = pd.MultiIndex.from_product([dates, modules], names=["date", "module"])
    out = df.reindex(index=full, columns=cols, fill_value=0).fillna(0)
    return out.to_numpy(dtype=np.float32).reshape(len(dates), len(modules), len(cols))


def build_daily(
    incidents: pd.DataFrame,
    signals: pd.DataFrame | None,
    calendar: pd.DataFrame,
    modules: list[str],
    rolling_windows: list[int] = (7, 14, 28),
    use_aux: bool = True,
    synthetic: bool = True,
) -> DailyData:
    dates = pd.DatetimeIndex(calendar["date"]).normalize()
    day = incidents["created_at"].dt.normalize()
    outside = ~day.isin(dates)
    if outside.any():
        raise ValueError(f"{int(outside.sum())} incidents fall outside the calendar range")

    per_prio = incidents.assign(date=day).groupby(["date", "module", "priority"]).size()
    by_prio = _grid(per_prio.unstack("priority"), dates, modules, PRIORITIES)  # [T, N, 4]
    counts = by_prio.sum(axis=2)
    p1p2 = by_prio[:, :, 0] + by_prio[:, :, 1]

    groups = {"counts": [f"n_{p}" for p in PRIORITIES] + ["n_total"]}
    blocks = [by_prio, counts[:, :, None]]
    if use_aux:
        if signals is None:
            raise ValueError("use_aux is true but no daily signals are available")
        sig = signals.assign(date=pd.to_datetime(signals["date"]).dt.normalize())
        blocks.append(_grid(sig.set_index(["date", "module"]), dates, modules, AUX_FEATURES))
        groups["aux"] = list(AUX_FEATURES)

    total = pd.DataFrame(counts)
    groups["rolling"] = [f"mean{w}_total" for w in rolling_windows]
    for w in rolling_windows:  # backward-looking: day d uses days d-w+1 .. d
        blocks.append(total.rolling(w, min_periods=1).mean().to_numpy(np.float32)[:, :, None])

    dow = dates.dayofweek.to_numpy()
    cal = np.column_stack(
        [
            calendar["is_weekend"].to_numpy(float),
            calendar["is_month_end_close"].to_numpy(float),
            calendar["is_quarter_end"].to_numpy(float),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
        ]
    ).astype(np.float32)
    groups["calendar"] = list(CALENDAR_FEATURES)
    blocks.append(np.repeat(cal[:, None, :], len(modules), axis=1))

    features = np.concatenate(blocks, axis=2).astype(np.float32)
    names = [n for g in groups.values() for n in g]
    assert features.shape[2] == len(names)
    return DailyData(dates, list(modules), names, groups, features, counts, p1p2, cal, synthetic)


# --------------------------------------------------------------------------- splits
@dataclass
class Split:
    """Day boundaries (indices into the daily axis) and forecast origins per part.

    Training period = days [0, train_end); validation = [train_end, val_end);
    test = [val_end, test_end). Origins are chosen so that all H target days of a sample
    fall inside its part; input windows may reach back into earlier parts (that history
    is known at forecast time)."""

    name: str
    train_end: int
    val_end: int
    test_end: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def describe(self, dates: pd.DatetimeIndex) -> dict:
        def period(a, b):
            return [str(dates[a].date()), str(dates[b - 1].date())]

        return {
            "name": self.name,
            "periods": {
                "train": period(0, self.train_end),
                "val": period(self.train_end, self.val_end),
                "test": period(self.val_end, self.test_end),
            },
            "n_origins": {"train": len(self.train), "val": len(self.val), "test": len(self.test)},
        }


def split_by_days(
    name: str, n_days: int, lookback: int, horizon: int, train_end: int, val_end: int, test_end: int
) -> Split:
    if not 0 < train_end < val_end < test_end <= n_days:
        raise ValueError("need 0 < train_end < val_end < test_end <= n_days")
    origins = np.arange(lookback - 1, n_days - horizon)  # full window, all targets exist
    first, last = origins + 1, origins + horizon  # first and last target day
    return Split(
        name,
        train_end,
        val_end,
        test_end,
        train=origins[last < train_end],
        val=origins[(first >= train_end) & (last < val_end)],
        test=origins[(first >= val_end) & (last < test_end)],
    )


def chronological_split(n_days: int, lookback: int, horizon: int, ratios: dict) -> Split:
    if abs(sum(ratios.values()) - 1) > 1e-9:
        raise ValueError("split ratios must sum to 1")
    train_end = round(n_days * ratios["train"])
    val_end = round(n_days * (ratios["train"] + ratios["val"]))
    return split_by_days("main", n_days, lookback, horizon, train_end, val_end, n_days)


def rolling_origin_splits(
    n_days: int, lookback: int, horizon: int, n_folds: int, val_days: int, test_days: int
) -> list[Split]:
    """Expanding-window folds whose test windows move forward and end at the last day."""
    folds = []
    for k in range(n_folds):
        test_end = n_days - (n_folds - 1 - k) * test_days
        val_end = test_end - test_days
        train_end = val_end - val_days
        folds.append(
            split_by_days(f"fold{k + 1}", n_days, lookback, horizon, train_end, val_end, test_end)
        )
    return folds


# --------------------------------------------------------------------------- scaling
@dataclass
class Scaler:
    """log1p then z-score for count-like features, fit on training-period days only."""

    index: list[int]  # feature positions that are scaled (calendar features are left as is)
    mean: list[float]
    std: list[float]
    fit_end: int  # fitted on days [0, fit_end)

    @classmethod
    def fit(cls, daily: DailyData, fit_end: int) -> Scaler:
        names = [n for g in SCALED_GROUPS for n in daily.groups.get(g, [])]
        idx = daily.feature_index(names)
        x = np.log1p(daily.features[:fit_end][:, :, idx]).reshape(-1, len(idx))
        std = x.std(axis=0)
        std[std < 1e-6] = 1.0  # constant feature: leave centred, avoid division by zero
        return cls(idx.tolist(), x.mean(axis=0).tolist(), std.tolist(), fit_end)

    def transform(self, features: np.ndarray) -> np.ndarray:
        out = features.astype(np.float32, copy=True)
        idx = np.asarray(self.index)
        out[..., idx] = (np.log1p(features[..., idx]) - np.asarray(self.mean)) / np.asarray(
            self.std
        )
        return out


# --------------------------------------------------------------------------- windows
def make_windows(
    features: np.ndarray,
    target: np.ndarray,
    future_calendar: np.ndarray,
    origins: np.ndarray,
    lookback: int,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Sliding windows for the given forecast origins (see module docstring for shapes)."""
    origins = np.asarray(origins, dtype=int)
    past = origins[:, None] + np.arange(-lookback + 1, 1)[None, :]  # [S, L]
    future = origins[:, None] + np.arange(1, horizon + 1)[None, :]  # [S, H]
    if len(origins) and (past.min() < 0 or future.max() >= len(features)):
        raise IndexError("window reaches outside the data")
    return {
        "X": features[past],  # [S, L, N, F]
        "Y": target[future],  # [S, H, N]
        "Xf": future_calendar[future],  # [S, H, C]
        "origins": origins,
    }


def prepare(
    cfg: dict,
    daily: DailyData,
    split: Split,
    target: str | None = None,
    lookback: int | None = None,
    drop_groups: tuple[str, ...] = (),
) -> dict:
    """Scaled train / val / test windows for one split (scaler fit on its training days).
    drop_groups removes whole feature groups, e.g. ("aux",) for the ablation without
    job failures / IDoc errors / change events."""
    ds = cfg["dataset"]
    target = target or ds["target"]
    lookback = lookback or ds["lookback"]
    horizon = max(ds["horizons"])
    scaler = Scaler.fit(daily, split.train_end)
    keep = [n for g, names in daily.groups.items() if g not in drop_groups for n in names]
    scaled = scaler.transform(daily.features)[:, :, daily.feature_index(keep)]
    y = daily.target(target)
    out = {
        "scaler": scaler,
        "split": split,
        "target": target,
        "lookback": lookback,
        "horizon": horizon,
        "feature_names": keep,
    }
    for part in ["train", "val", "test"]:
        out[part] = make_windows(
            scaled, y, daily.future_calendar, getattr(split, part), lookback, horizon
        )
    return out


# --------------------------------------------------------------------------- tabular (LightGBM)
def tabular_features(
    daily: DailyData, origins: np.ndarray, h: int, lags: list[int], target: str = "regression"
) -> pd.DataFrame:
    """One row per (forecast origin, module) for a model that needs flat features.

    Only data up to the origin day is used, plus the calendar of the target day t+h
    (known in advance). Lags that reach before the first day are NaN (LightGBM handles NaN).
    """
    origins = np.asarray(origins, dtype=int)
    n_mod = len(daily.modules)
    t = np.repeat(origins, n_mod)  # origin day per row
    m = np.tile(np.arange(n_mod), len(origins))  # module per row
    feats, names = daily.features, daily.feature_names

    def at(series: np.ndarray, days: np.ndarray) -> np.ndarray:
        out = np.full(len(days), np.nan, dtype=np.float32)
        ok = days >= 0
        out[ok] = series[days[ok], m[ok]]
        return out

    cols = {
        "origin": daily.dates[t],
        "module": pd.Categorical([daily.modules[i] for i in m], categories=daily.modules),
        "horizon": h,
    }
    for k in lags:
        cols[f"lag{k}_total"] = at(daily.counts, t - k)
    cols["p1p2_last7"] = sum(np.nan_to_num(at(daily.p1p2, t - k)) for k in range(7))
    for n in daily.groups["counts"][:4] + daily.groups["rolling"] + daily.groups.get("aux", []):
        cols[n] = feats[t, m, names.index(n)]
    if "aux" in daily.groups:
        cols["changes_last3"] = sum(
            np.nan_to_num(at(feats[:, :, names.index("change_events")], t - k)) for k in range(3)
        )
    target_day = t + h
    for j, c in enumerate(CALENDAR_FEATURES[:3]):
        cols[f"target_{c}"] = daily.future_calendar[target_day, j]
    cols["target_dow"] = daily.dates[target_day].dayofweek
    cols["y"] = daily.target(target)[target_day, m]
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------- saving
def save_daily(daily: DailyData, path) -> None:
    np.savez_compressed(
        path,
        features=daily.features,
        counts=daily.counts,
        p1p2=daily.p1p2,
        future_calendar=daily.future_calendar,
        # plain fixed-width strings: object arrays would need pickle to load (unsafe)
        dates=np.array(daily.dates.strftime("%Y-%m-%d").tolist(), dtype="U10"),
        modules=np.array(daily.modules, dtype=str),
        feature_names=np.array(daily.feature_names, dtype=str),
        groups=json.dumps(daily.groups),
        synthetic=daily.synthetic,
    )


def load_daily(cfg: dict) -> DailyData:
    z = np.load(get_path(cfg, "processed") / "daily.npz")
    return DailyData(
        pd.DatetimeIndex(pd.to_datetime(z["dates"])),
        z["modules"].tolist(),
        z["feature_names"].tolist(),
        json.loads(str(z["groups"])),
        z["features"],
        z["counts"],
        z["p1p2"],
        z["future_calendar"],
        bool(z["synthetic"]),
    )


def plot_splits(daily: DailyData, main: Split, folds: list[Split], path) -> None:
    d = daily.dates
    fig, ax = plt.subplots(figsize=(12, 3.2))
    colors = {"train": "#9db4cc", "val": "#f2b36b", "test": "#d9534f"}
    for row, s in enumerate([main, *folds]):
        for part, a, b in [
            ("train", 0, s.train_end),
            ("val", s.train_end, s.val_end),
            ("test", s.val_end, s.test_end),
        ]:
            ax.barh(
                row,
                (d[b - 1] - d[a]).days + 1,
                left=d[a],
                color=colors[part],
                label=part if row == 0 else None,
            )
    ax.set_yticks(range(len(folds) + 1), ["main split"] + [f.name for f in folds])
    ax.invert_yaxis()
    ax.legend(loc="lower left", ncol=3, fontsize=8)
    ax.set_title(
        ("SYNTHETIC DATA - " if daily.synthetic else "")
        + "chronological split and rolling-origin folds (no shuffling)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    ds = cfg["dataset"]
    incidents, signals, calendar, synthetic = load_source(cfg)
    daily = build_daily(
        incidents,
        signals,
        calendar,
        list(cfg["data"]["modules"]),
        ds["rolling_windows"],
        ds["use_aux"],
        synthetic,
    )
    t_days, n_mod, n_feat = daily.shape
    lookback, horizon = ds["lookback"], max(ds["horizons"])
    main_split = chronological_split(t_days, lookback, horizon, ds["split"])
    ro = ds["rolling_origin"]
    folds = rolling_origin_splits(
        t_days, lookback, horizon, ro["n_folds"], ro["val_days"], ro["test_days"]
    )
    data = prepare(cfg, daily, main_split)

    out = get_path(cfg, "processed")
    save_daily(daily, out / "daily.npz")
    splits = {
        "main": main_split.describe(daily.dates),
        "rolling_origin": [f.describe(daily.dates) for f in folds],
        "boundaries": {
            s.name: {"train_end": s.train_end, "val_end": s.val_end, "test_end": s.test_end}
            for s in [main_split, *folds]
        },
    }
    with open(out / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    with open(out / "scaler_main.json", "w", encoding="utf-8") as f:
        json.dump(asdict(data["scaler"]), f, indent=2)
    plot_splits(daily, main_split, folds, get_path(cfg, "figures") / "time_split.png")

    y_cls = daily.target("classification")
    summary = {
        "synthetic": synthetic,
        "daily_tensor": {"days": t_days, "modules": n_mod, "features": n_feat},
        "feature_groups": daily.groups,
        "lookback": lookback,
        "horizon_max": horizon,
        "reported_horizons": ds["horizons"],
        "target": ds["target"],
        "splits": splits["main"],
        "rolling_origin": splits["rolling_origin"],
        "window_shapes": {
            p: {k: list(v.shape) for k, v in data[p].items() if k != "origins"}
            for p in ["train", "val", "test"]
        },
        "target_stats": {
            p: {
                "mean_count_per_module_day": round(float(daily.counts[a:b].mean()), 4),
                "share_module_days_with_p1p2": round(float(y_cls[a:b].mean()), 4),
            }
            for p, a, b in [
                ("train", 0, main_split.train_end),
                ("val", main_split.train_end, main_split.val_end),
                ("test", main_split.val_end, main_split.test_end),
            ]
        },
    }
    with open(get_path(cfg, "results") / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    label = "SYNTHETIC DATA" if synthetic else "REAL DATA"
    print(f"{label}: daily tensor [T={t_days}, N={n_mod}, F={n_feat}] (days, modules, features)")
    for g, names in daily.groups.items():
        print(f"  {g:<9} {names}")
    print(
        f"look-back L={lookback}, forecast days 1..{horizon} (reported: {ds['horizons']}), "
        f"target={ds['target']}"
    )
    for p in ["train", "val", "test"]:
        w = data[p]
        print(
            f"  {p:<5} {splits['main']['periods'][p][0]} .. {splits['main']['periods'][p][1]}"
            f"  X {list(w['X'].shape)}  Y {list(w['Y'].shape)}  Xf {list(w['Xf'].shape)}"
        )
    for f in splits["rolling_origin"]:
        print(
            f"  {f['name']}: train to {f['periods']['train'][1]}, val {f['periods']['val'][0]}"
            f" .., test {f['periods']['test'][0]} .. {f['periods']['test'][1]}"
        )
    print(
        "saved data/processed/{daily.npz, splits.json, scaler_main.json}, "
        "results/dataset_summary.json, results/figures/time_split.png"
    )


if __name__ == "__main__":
    main()
