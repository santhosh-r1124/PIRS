"""Dataset: shapes, alignment of windows, chronological splits and NO LEAKAGE."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.data.build_dataset import (
    Scaler,
    build_daily,
    chronological_split,
    load_daily,
    load_source,
    prepare,
    rolling_origin_splits,
    save_daily,
    tabular_features,
)
from src.data.generate_synthetic import generate
from src.utils import load_config

L, H = 28, 7


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def raw(cfg):
    d = generate(cfg, seed=42)
    return d.incidents, d.signals, d.calendar, d.counts


def daily_from(cfg, incidents, signals, calendar, use_aux=True):
    return build_daily(incidents, signals, calendar, cfg["data"]["modules"],
                       cfg["dataset"]["rolling_windows"], use_aux)  # fmt: skip


@pytest.fixture(scope="module")
def daily(cfg, raw):
    return daily_from(cfg, *raw[:3])


@pytest.fixture(scope="module")
def split(daily, cfg):
    return chronological_split(len(daily.dates), L, H, cfg["dataset"]["split"])


# --------------------------------------------------------------------------- tensor
def test_daily_tensor_shape(daily):
    assert daily.shape == (1096, 10, 16)
    assert daily.feature_names == [n for g in daily.groups.values() for n in g]
    assert list(daily.groups) == ["counts", "aux", "rolling", "calendar"]
    cal = daily.features[:, :, daily.feature_index(daily.groups["calendar"])]
    assert np.all(cal == cal[:, :1, :])  # calendar is the same for every module


def test_aggregation_matches_raw_data(daily, raw):
    _, signals, _, counts = raw
    np.testing.assert_array_equal(daily.counts, counts.to_numpy())
    by_prio = daily.features[:, :, daily.feature_index(["n_P1", "n_P2", "n_P3", "n_P4"])]
    np.testing.assert_array_equal(by_prio.sum(axis=2), daily.counts)
    np.testing.assert_array_equal(daily.p1p2, by_prio[:, :, 0] + by_prio[:, :, 1])
    jf = signals.pivot(index="date", columns="module", values="job_failures")[daily.modules]
    np.testing.assert_array_equal(daily.features[:, :, daily.feature_index(["job_failures"])[0]],
                                  jf.to_numpy())  # fmt: skip


def test_features_use_only_past_days(cfg, raw, daily):
    incidents, signals, calendar, _ = raw
    d = 500
    extra = incidents[incidents["module"] == "FI"].head(20).copy()
    extra["incident_id"] = [f"X{i}" for i in range(len(extra))]
    extra["created_at"] = daily.dates[d] + pd.Timedelta(hours=10)
    changed = daily_from(cfg, pd.concat([incidents, extra]), signals, calendar)
    np.testing.assert_array_equal(changed.features[:d], daily.features[:d])
    assert not np.array_equal(changed.features[d], daily.features[d])


# --------------------------------------------------------------------------- splits
def check_split(s):
    for part, start, end in [("train", 0, s.train_end), ("val", s.train_end, s.val_end),
                             ("test", s.val_end, s.test_end)]:  # fmt: skip
        o = getattr(s, part)
        assert len(o) > 0, part
        assert np.all(np.diff(o) == 1), part  # consecutive, in time order
        assert o.min() >= L - 1  # a full look-back window exists
        assert o.min() + 1 >= start and o.max() + H < end, part  # all targets inside the part
    target_days = [set((getattr(s, p)[:, None] + np.arange(1, H + 1)).ravel())
                   for p in ["train", "val", "test"]]  # fmt: skip
    assert not (target_days[0] & target_days[1]) and not (target_days[1] & target_days[2])


def test_chronological_split(split, daily):
    check_split(split)
    t_days = len(daily.dates)
    assert split.train_end == round(0.70 * t_days) and split.val_end == round(0.85 * t_days)
    assert split.test_end == t_days


def test_rolling_origin_folds_move_forward(daily, cfg):
    ro = cfg["dataset"]["rolling_origin"]
    folds = rolling_origin_splits(len(daily.dates), L, H, ro["n_folds"], ro["val_days"],
                                  ro["test_days"])  # fmt: skip
    assert len(folds) == 4
    for f in folds:
        check_split(f)
    assert folds[-1].test_end == len(daily.dates)
    for a, b in zip(folds, folds[1:], strict=False):
        assert b.train_end > a.train_end and b.val_end == a.test_end  # windows move forward


# --------------------------------------------------------------------------- windows
def test_window_alignment(cfg, daily, split):
    data = prepare(cfg, daily, split)
    scaled = data["scaler"].transform(daily.features)
    for part in ["train", "val", "test"]:
        w = data[part]
        assert w["X"].shape[1:] == (L, 10, 16) and w["Y"].shape[1:] == (H, 10)
        assert w["Xf"].shape[1:] == (H, 5)
        s = len(w["origins"]) // 2
        t = w["origins"][s]
        np.testing.assert_array_equal(w["X"][s, -1], scaled[t])  # last input day = origin
        np.testing.assert_array_equal(w["X"][s, 0], scaled[t - L + 1])
        for h in [1, 3, 7]:
            np.testing.assert_array_equal(w["Y"][s, h - 1], daily.counts[t + h])
            np.testing.assert_array_equal(w["Xf"][s, h - 1], daily.future_calendar[t + h])


def test_shorter_lookback(cfg, daily):
    s14 = chronological_split(len(daily.dates), 14, H, cfg["dataset"]["split"])
    data = prepare(cfg, daily, s14, lookback=14)
    assert data["train"]["X"].shape[1] == 14
    assert data["train"]["origins"][0] == 13


def test_classification_target(cfg, daily, split):
    data = prepare(cfg, daily, split, target="classification")
    y = data["train"]["Y"]
    assert set(np.unique(y)) == {0.0, 1.0}
    t = data["train"]["origins"][0]
    np.testing.assert_array_equal(y[0, 0], (daily.p1p2[t + 1] > 0).astype(float))


# --------------------------------------------------------------------------- leakage
def perturb_after(incidents, signals, start):
    """Make everything from `start` onwards wildly different."""
    late = incidents[incidents["created_at"] >= start]
    copies = []
    for k in range(3):
        c = late.copy()
        c["incident_id"] = c["incident_id"] + f"_dup{k}"
        c["priority"] = "P1"
        copies.append(c)
    sig = signals.copy()
    num = ["job_failures", "idoc_errors", "change_events"]
    sig.loc[sig["date"] >= start, num] = sig.loc[sig["date"] >= start, num] * 10 + 5
    return pd.concat([incidents, *copies], ignore_index=True), sig


def test_no_test_or_validation_data_reaches_training(cfg, raw, daily, split):
    incidents, signals, calendar, _ = raw
    start = daily.dates[split.train_end]  # first validation day
    bad_inc, bad_sig = perturb_after(incidents, signals, start)
    bad = daily_from(cfg, bad_inc, bad_sig, calendar)

    clean_data, bad_data = prepare(cfg, daily, split), prepare(cfg, bad, split)
    # 1. scaler statistics come from the training period only
    assert clean_data["scaler"] == bad_data["scaler"]
    # 2. training inputs and targets are bit-for-bit identical
    for k in ["X", "Y", "Xf"]:
        np.testing.assert_array_equal(clean_data["train"][k], bad_data["train"][k])
    # 3. sanity check: the perturbation really changed the later data
    assert not np.array_equal(clean_data["test"]["Y"], bad_data["test"]["Y"])
    assert bad.counts[split.train_end :].sum() == 4 * daily.counts[split.train_end :].sum()


def test_scaler_fit_on_training_days(daily, split):
    sc = Scaler.fit(daily, split.train_end)
    scaled = sc.transform(daily.features)
    train = scaled[: split.train_end][:, :, sc.index].reshape(-1, len(sc.index))
    np.testing.assert_allclose(train.mean(axis=0), 0, atol=1e-4)
    np.testing.assert_allclose(train.std(axis=0), 1, atol=1e-3)
    cal = daily.feature_index(daily.groups["calendar"])
    assert not set(cal) & set(sc.index)  # calendar features are not rescaled
    np.testing.assert_array_equal(scaled[:, :, cal], daily.features[:, :, cal])


# --------------------------------------------------------------------------- tabular + options
def test_tabular_features(cfg, daily, split):
    lags = cfg["dataset"]["tabular_lags"]
    origins = split.train[:50]
    tab = tabular_features(daily, origins, h=3, lags=lags)
    assert len(tab) == 50 * 10
    m = daily.modules.index("SD")
    row = tab[(tab["origin"] == daily.dates[origins[10]]) & (tab["module"] == "SD")].iloc[0]
    t = origins[10]
    assert row["lag0_total"] == daily.counts[t, m] and row["lag27_total"] == daily.counts[t - 27, m]
    assert row["y"] == daily.counts[t + 3, m]
    assert row["target_is_month_end_close"] == daily.future_calendar[t + 3, 1]
    early = tabular_features(daily, np.array([13]), h=1, lags=lags)
    assert early["lag27_total"].isna().all()  # before the first day -> NaN, not wrapped around


def test_tabular_features_ignore_the_future(cfg, raw, daily):
    incidents, signals, calendar, _ = raw
    t = 600
    extra = incidents.head(30).copy()
    extra["incident_id"] = [f"F{i}" for i in range(30)]
    extra["created_at"] = daily.dates[t + 1] + pd.Timedelta(hours=9)
    later = daily_from(cfg, pd.concat([incidents, extra]), signals, calendar)
    a = tabular_features(daily, np.array([t]), 1, cfg["dataset"]["tabular_lags"])
    b = tabular_features(later, np.array([t]), 1, cfg["dataset"]["tabular_lags"])
    pd.testing.assert_frame_equal(a.drop(columns="y"), b.drop(columns="y"))
    assert not a["y"].equals(b["y"])


def test_without_aux_signals(cfg, raw):
    incidents, _, calendar, _ = raw
    d = daily_from(cfg, incidents, None, calendar, use_aux=False)
    assert d.shape[2] == 13 and "aux" not in d.groups
    tab = tabular_features(d, np.array([100]), 1, [0, 1])
    assert "job_failures" not in tab.columns and "changes_last3" not in tab.columns


def test_real_source_without_export_fails_loudly(cfg, tmp_path):
    c = copy.deepcopy(cfg)
    c["data"]["source"] = "real"
    c["real_data"]["incidents_csv"] = str(tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError):
        load_source(c)


def test_save_and_load_roundtrip(cfg, daily, tmp_path):
    c = copy.deepcopy(cfg)
    c["paths"]["processed"] = str(tmp_path)
    save_daily(daily, tmp_path / "daily.npz")
    back = load_daily(c)
    np.testing.assert_array_equal(back.features, daily.features)
    assert back.dates.equals(daily.dates) and back.groups == daily.groups
    assert back.synthetic is True
