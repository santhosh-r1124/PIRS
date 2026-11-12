"""Synthetic generator: valid, reproducible, and the built-in mechanisms are really there."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.data import text_templates as tt
from src.data.generate_synthetic import generate, propagation_matrix, summarise, validate
from src.utils import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def data(cfg):
    return generate(cfg, seed=42)


@pytest.fixture(scope="module")
def data_no_prop(cfg):
    return generate(cfg, seed=42, propagation=False)


def small_cfg(cfg, n_days=120):
    c = copy.deepcopy(cfg)
    c["synthetic"]["n_days"] = n_days
    return c


def test_output_matches_schema(data, cfg):
    for rep in validate(data, cfg):
        assert rep.ok, rep
    assert len(data.signals) == len(data.calendar) * len(cfg["data"]["modules"])
    daily = data.incidents.groupby([data.incidents["created_at"].dt.normalize(), "module"]).size()
    assert daily.sum() == data.counts.to_numpy().sum()


def test_reproducible(cfg):
    c = small_cfg(cfg)
    a, b, other = generate(c, seed=7), generate(c, seed=7), generate(c, seed=8)
    pd.testing.assert_frame_equal(a.incidents, b.incidents)
    pd.testing.assert_frame_equal(a.signals, b.signals)
    assert not a.counts.equals(other.counts)


def weekday_masks(data):
    weekday = ~data.calendar["is_weekend"].to_numpy()
    close = data.calendar["is_month_end_close"].to_numpy()
    return weekday, close


def test_month_end_close_raises_rate(data, cfg):
    weekday, close = weekday_masks(data)
    for m in cfg["synthetic"]["close_factor"]:  # FI, CO, SD
        c = data.counts[m]
        ratio = c[weekday & close].mean() / c[weekday & ~close].mean()
        assert ratio > 1.2, (m, ratio)


def test_weekend_lowers_rate(data):
    weekday, _ = weekday_masks(data)
    for m in ["FI", "CO", "SD", "MM", "HCM"]:
        c = data.counts[m]
        assert c[~weekday].mean() < 0.5 * c[weekday].mean(), m


def test_propagation_increases_downstream_incidents(data, data_no_prop, cfg):
    modules = cfg["data"]["modules"]
    w = propagation_matrix(cfg["synthetic"]["propagation"], modules)
    on, off = data.counts, data_no_prop.counts
    # Separate random streams: switching propagation off changes nothing else.
    assert (on >= off).all().all()
    no_parents = [m for j, m in enumerate(modules) if w[:, j].sum() == 0]
    assert no_parents == ["BASIS"]
    pd.testing.assert_series_equal(on["BASIS"], off["BASIS"])
    for m in ["FI", "CO", "MM", "WM", "PP"]:
        assert on[m].sum() > 1.2 * off[m].sum(), m
    n_prop = (data.causes["cause"] == "propagation").sum()
    assert on.to_numpy().sum() - off.to_numpy().sum() == n_prop


def test_ripples_only_along_true_edges(data, cfg):
    modules = cfg["data"]["modules"]
    w = propagation_matrix(cfg["synthetic"]["propagation"], modules)
    true_pairs = {(modules[i], modules[j]) for i, j in zip(*np.nonzero(w), strict=True)}
    caused = data.incidents.merge(data.causes, on="incident_id")
    caused = caused[caused["cause"] == "propagation"]
    seen = set(zip(caused["source_module"], caused["module"], strict=True))
    assert seen <= true_pairs
    assert ("QM", "MM") not in seen  # in the hand-built graph, but strength 0
    assert {("SD", "WM"), ("MM", "PP")} <= seen  # real ripples missing from the graph


def test_ground_truth_record(data):
    gt = data.ground_truth
    assert gt["synthetic"] is True
    assert gt["true_edges_missing_from_hand_built_graph"] == [["MM", "PP"], ["SD", "WM"]]
    assert gt["hand_built_edges_without_true_effect"] == [["QM", "MM"]]
    assert gt["spectral_radius"] < 1  # the cascade process is stable (no explosion)
    w = np.array(gt["matrix_source_to_target"])
    i = {m: k for k, m in enumerate(gt["modules"])}
    assert w[i["SD"], i["FI"]] == pytest.approx(0.30)


def test_aux_signals_track_incidents(data, cfg):
    s = summarise(data, cfg)
    assert s["aux_correlation_with_same_day_incidents"]["job_failures"] > 0.05
    assert s["aux_correlation_with_same_day_incidents"]["idoc_errors"] > 0.1
    changes = data.signals["change_events"]
    assert set(changes.unique()) <= {0, 1} and changes.sum() > 0


def test_ticket_text_is_varied_and_consistent(data):
    inc = data.incidents
    assert inc["short_description"].nunique() > 0.4 * len(inc)
    for cat, group in inc.groupby("category"):
        assert group["short_description"].nunique() >= 30, cat
    # The job named in a job-failure description is the job restarted in the notes.
    jobs = [j for v in tt.VOCAB.values() for j in v["job"]]
    jf = inc[inc["category"] == "job_failure"]
    checked = 0
    for desc, notes in zip(jf["short_description"], jf["resolution_notes"], strict=True):
        named = [j for j in jobs if j in desc]
        if named:
            assert named[0] in notes, (desc, notes)
            checked += 1
    assert checked > 100


def test_categories_match_config(cfg):
    assert cfg["data"]["categories"] == tt.CATEGORIES
    assert set(tt.CATEGORY_MIX) == set(cfg["data"]["modules"])
