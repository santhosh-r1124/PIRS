"""Baselines. A deep model only counts as an improvement if it beats these on the same split.

Every forecaster returns predictions shaped like the dataset targets, [S, H, N]
(S forecast origins, H forecast days, N modules), for the target series y:
  regression      y = incident count per module and day
  classification  y = 1 if the module had at least one P1/P2 incident that day

  naive            y(t+h) = y(t)                       "tomorrow = today"
  seasonal naive   y(t+h) = y(t+h-7)                   same weekday last week
  historical mean  mean of y over the TRAINING period for (module, weekday of t+h)
  LightGBM         one global gradient-boosted model per forecast day h, rows = (origin, module),
                   lag / rolling / calendar features from build_dataset.tabular_features,
                   module as a categorical feature; Poisson objective for counts, binary for P1/P2
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.build_dataset import DailyData, Split, tabular_features
from src.metrics import score

NOT_FEATURES = ["origin", "horizon", "y"]


def target_windows(y: np.ndarray, origins: np.ndarray, horizon: int) -> np.ndarray:
    """[S, H, N] true values for days t+1..t+H."""
    return y[np.asarray(origins)[:, None] + np.arange(1, horizon + 1)[None, :]]


def naive(y: np.ndarray, origins: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(y[np.asarray(origins)][:, None, :], horizon, axis=1)


def seasonal_naive(y: np.ndarray, origins: np.ndarray, horizon: int, season: int = 7):
    h = np.arange(1, horizon + 1)
    back = season * np.ceil(h / season).astype(int)  # 7 for h <= 7: never looks past t
    return y[np.asarray(origins)[:, None] + h[None, :] - back[None, :]]


class HistoricalMean:
    """Mean of y per (weekday, module) over training days [0, train_end) only."""

    def fit(self, dates: pd.DatetimeIndex, y: np.ndarray, train_end: int) -> HistoricalMean:
        dow = np.asarray(dates.dayofweek)[:train_end]
        self.table = np.stack([y[:train_end][dow == d].mean(axis=0) for d in range(7)])
        self.dow = np.asarray(dates.dayofweek)
        return self

    def predict(self, origins: np.ndarray, horizon: int) -> np.ndarray:
        idx = np.asarray(origins)[:, None] + np.arange(1, horizon + 1)[None, :]
        return self.table[self.dow[idx]]  # [S, H, N]


def lightgbm_forecast(
    daily: DailyData,
    split: Split,
    target: str,
    horizon: int,
    params: dict,
    lags: list[int],
    seed: int,
) -> tuple[dict, dict, pd.DataFrame]:
    """Fit one model per forecast day on training rows, early-stop on validation rows.

    Returns ({"val": [S, H, N], "test": [S, H, N]}, best iterations, gain importance)."""
    import lightgbm as lgb  # imported here so the other baselines run without it

    params = dict(params)
    rounds = params.pop("early_stopping_rounds")
    n_mod = len(daily.modules)
    preds = {p: np.zeros((len(getattr(split, p)), horizon, n_mod)) for p in ["val", "test"]}
    best, importance = {}, []
    for h in range(1, horizon + 1):
        rows = {
            p: tabular_features(daily, getattr(split, p), h, lags, target)
            for p in ["train", "val", "test"]
        }
        x = {p: df.drop(columns=NOT_FEATURES) for p, df in rows.items()}
        common = dict(random_state=seed, n_jobs=1, verbose=-1, **params)
        if target == "regression":
            model, metric = lgb.LGBMRegressor(objective="poisson", **common), "poisson"
        else:
            model, metric = lgb.LGBMClassifier(objective="binary", **common), "binary_logloss"
        model.fit(
            x["train"],
            rows["train"]["y"],
            eval_set=[(x["val"], rows["val"]["y"])],
            eval_metric=metric,
            callbacks=[lgb.early_stopping(rounds, verbose=False)],
        )
        best[f"h{h}"] = int(model.best_iteration_)
        for p in ["val", "test"]:
            out = model.predict(x[p]) if target == "regression" else model.predict_proba(x[p])[:, 1]
            preds[p][:, h - 1, :] = np.asarray(out).reshape(-1, n_mod)  # rows: origin-major
        importance.append(
            pd.DataFrame(
                {
                    "horizon": h,
                    "feature": model.feature_name_,
                    "gain": model.booster_.feature_importance(importance_type="gain"),
                }
            )
        )
    return preds, best, pd.concat(importance, ignore_index=True)


def run_baselines(cfg: dict, daily: DailyData, split: Split, target: str, seed: int) -> dict:
    """Predictions and metrics of all baselines on the validation and test splits."""
    horizons = cfg["dataset"]["horizons"]
    horizon = max(horizons)
    ev = cfg["evaluation"]
    y = daily.target(target)
    parts = ["val", "test"]
    truth = {p: target_windows(y, getattr(split, p), horizon) for p in parts}

    hist = HistoricalMean().fit(daily.dates, y, split.train_end)
    preds = {
        "naive": {p: naive(y, getattr(split, p), horizon) for p in parts},
        "seasonal_naive": {p: seasonal_naive(y, getattr(split, p), horizon) for p in parts},
        "historical_mean": {p: hist.predict(getattr(split, p), horizon) for p in parts},
    }
    lgbm_info, importance = {"status": "not run"}, None
    try:
        lgbm_preds, best, importance = lightgbm_forecast(
            daily,
            split,
            target,
            horizon,
            cfg["baselines"]["lightgbm"],
            cfg["dataset"]["tabular_lags"],
            seed,
        )
        preds["lightgbm"] = lgbm_preds
        lgbm_info = {
            "status": "ok",
            "best_iteration": best,
            "note": "trees chosen by early stopping on val: val scores are optimistic",
        }
    except ImportError as e:
        lgbm_info = {"status": f"not run: {e}"}

    metrics = {p: {} for p in parts}
    for name, pr in preds.items():
        scored = score(target, truth, pr, horizons, ev["precision_at_k"], ev["deviance_floor"])
        for p in parts:
            metrics[p][name] = scored[p]
    return {
        "metrics": metrics,
        "preds": preds,
        "truth": truth,
        "lightgbm": lgbm_info,
        "importance": importance,
    }
