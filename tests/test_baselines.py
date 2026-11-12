"""Baselines: correct alignment, no look-ahead, training-only statistics, LightGBM wiring."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.data.build_dataset import build_daily, chronological_split, tabular_features
from src.data.generate_synthetic import generate
from src.models.baselines import (
    HistoricalMean,
    naive,
    run_baselines,
    seasonal_naive,
    target_windows,
)
from src.train import metrics_table, to_markdown
from src.utils import load_config

H = 7


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def daily(cfg):
    d = generate(cfg, seed=42)
    return build_daily(
        d.incidents,
        d.signals,
        d.calendar,
        cfg["data"]["modules"],
        cfg["dataset"]["rolling_windows"],
    )


@pytest.fixture(scope="module")
def split(cfg, daily):
    return chronological_split(
        len(daily.dates), cfg["dataset"]["lookback"], H, cfg["dataset"]["split"]
    )


def test_naive_and_seasonal_naive_alignment(daily, split):
    y, o = daily.counts, split.test
    nv, sn, truth = naive(y, o, H), seasonal_naive(y, o, H), target_windows(y, o, H)
    assert nv.shape == sn.shape == truth.shape == (len(o), H, 10)
    s, t = 5, o[5]
    for h in range(1, H + 1):
        np.testing.assert_array_equal(nv[s, h - 1], y[t])  # tomorrow = today
        np.testing.assert_array_equal(sn[s, h - 1], y[t + h - 7])  # same weekday last week
        np.testing.assert_array_equal(truth[s, h - 1], y[t + h])


def test_seasonal_naive_never_looks_ahead(daily, split):
    future = daily.counts.copy()
    future[split.test[0] + 1 :] = 99  # everything after the first test origin
    a = seasonal_naive(daily.counts, split.test[:1], H)
    b = seasonal_naive(future, split.test[:1], H)
    np.testing.assert_array_equal(a, b)


def test_historical_mean_uses_training_days_only(daily, split):
    y = daily.counts.astype(float)
    hm = HistoricalMean().fit(daily.dates, y, split.train_end)
    changed = y.copy()
    changed[split.train_end :] *= 10
    hm2 = HistoricalMean().fit(daily.dates, changed, split.train_end)
    np.testing.assert_array_equal(hm.table, hm2.table)
    dow = daily.dates[: split.train_end].dayofweek
    manual = pd.DataFrame(y[: split.train_end]).groupby(np.asarray(dow)).mean().to_numpy()
    np.testing.assert_allclose(hm.table, manual)
    pred = hm.predict(split.test[:3], H)
    t = split.test[0]
    np.testing.assert_allclose(pred[0, 0], hm.table[daily.dates[t + 1].dayofweek])


def test_tabular_rows_reshape_to_origin_by_module(daily, split):
    """LightGBM predictions are reshaped with reshape(-1, N); rows must be origin-major."""
    tab = tabular_features(daily, split.val, 1, [0])
    np.testing.assert_array_equal(
        tab["lag0_total"].to_numpy().reshape(-1, 10), daily.counts[split.val]
    )


@pytest.mark.parametrize("target", ["regression", "classification"])
def test_run_baselines_structure(cfg, daily, split, target):
    c = copy.deepcopy(cfg)
    c["baselines"]["lightgbm"]["n_estimators"] = 30  # keep the test fast where LightGBM exists
    res = run_baselines(c, daily, split, target, seed=42)
    simple = {"naive", "seasonal_naive", "historical_mean"}
    assert simple <= set(res["metrics"]["test"])
    for part in ["val", "test"]:
        for name, m in res["metrics"][part].items():
            assert set(m) >= {"h1", "h3", "h7"}, name
            assert all(np.isfinite(v) for v in m["h1"].values() if isinstance(v, float)), name
    for name, by_part in res["preds"].items():
        assert by_part["test"].shape == (len(split.test), H, 10), name
    if res["lightgbm"]["status"] == "ok":
        p = res["preds"]["lightgbm"]["test"]
        assert np.all(p >= 0) and (target == "regression" or np.all(p <= 1))
    else:
        assert res["lightgbm"]["status"].startswith("not run")


def test_lightgbm_trains_and_predicts(cfg, daily, split):
    pytest.importorskip("lightgbm")  # the environment test already fails if it is missing
    from src.models.baselines import lightgbm_forecast

    params = dict(cfg["baselines"]["lightgbm"], n_estimators=50, early_stopping_rounds=10)
    for target in ["regression", "classification"]:
        preds, best, imp = lightgbm_forecast(
            daily, split, target, 2, params, cfg["dataset"]["tabular_lags"], seed=42
        )
        assert preds["test"].shape == (len(split.test), 2, 10)
        assert np.all(np.isfinite(preds["test"])) and np.all(preds["test"] >= 0)
        assert all(1 <= b <= 50 for b in best.values())
        assert {"module", "lag0_total", "target_is_month_end_close"} <= set(imp["feature"])


def test_table_helpers():
    metrics = {"test": {"naive": {"h1": {"MAE": 1.0, "RMSE": 2.0, "PoissonDev": 3.0, "P@3": 0.5}}}}
    t = metrics_table(metrics, "regression", [1])
    assert list(t.columns) == ["split", "model", "MAE@1", "RMSE@1", "PoissonDev@1", "P@3@1"]
    md = to_markdown(t)
    assert md.splitlines()[0].startswith("| split | model") and "1.000" in md
