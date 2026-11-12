"""Metrics: hand-checked values, tie handling in precision@k / recall@k."""

import numpy as np
import pytest
from sklearn.metrics import mean_poisson_deviance

from src.metrics import (
    classification_metrics,
    f1_threshold,
    poisson_deviance,
    precision_at_k,
    recall_at_k,
    regression_metrics,
    top_k_hits,
)


def test_regression_metrics_by_hand():
    y = np.array([[[1, 0, 2], [0, 0, 0]], [[3, 1, 0], [1, 1, 1]]], dtype=float)  # [S=2, H=2, N=3]
    pred = y.copy()
    pred[:, 0, :] += 1  # h=1 off by exactly 1 everywhere
    m = regression_metrics(y, pred, horizons=[1, 2], k=1)
    assert m["h1"]["MAE"] == pytest.approx(1.0) and m["h1"]["RMSE"] == pytest.approx(1.0)
    assert m["h2"]["MAE"] == 0 and m["h2"]["RMSE"] == 0


def test_poisson_deviance():
    y = np.array([0.0, 1.0, 3.0, 2.0])
    mu = np.array([0.5, 1.5, 2.0, 2.0])
    assert poisson_deviance(y, mu) == pytest.approx(mean_poisson_deviance(y, mu))
    assert poisson_deviance(y[1:], y[1:]) == pytest.approx(0.0)  # perfect forecast
    assert np.isfinite(poisson_deviance(np.array([2.0]), np.array([0.0])))  # floored, not inf


def test_top_k_with_and_without_ties():
    event = np.array([[1, 0, 1, 0, 0], [0, 0, 0, 1, 1]], dtype=float)
    perfect = event + 0.1 * np.arange(5)  # events ranked first
    assert precision_at_k(event, perfect, k=2) == 1.0
    assert recall_at_k(event, perfect, k=2) == 1.0
    flat = np.zeros_like(event)  # all tied: expectation = random choice of 2 out of 5
    np.testing.assert_allclose(top_k_hits(event, flat, k=2), [0.8, 0.8])
    assert precision_at_k(event, flat, k=2) == pytest.approx(0.4)
    worst = -perfect
    assert precision_at_k(event, worst, k=2) == 0.0


def test_classification_metrics():
    rng = np.random.default_rng(0)
    y = (rng.random((50, 1, 10)) < 0.2).astype(float)
    perfect = y + rng.random(y.shape) * 0.1
    thr = {"h1": f1_threshold(y[:, 0], perfect[:, 0])}
    m = classification_metrics(y, perfect, [1], thr)["h1"]
    assert m["PR_AUC"] == pytest.approx(1.0) and m["ROC_AUC"] == pytest.approx(1.0)
    assert m["recall_at_threshold"] == 1.0 and m["precision_at_threshold"] == 1.0
    assert m["positive_rate"] == pytest.approx(y.mean())
    noise = rng.random(y.shape)
    thr = {"h1": f1_threshold(y[:, 0], noise[:, 0])}
    assert classification_metrics(y, noise, [1], thr)["h1"]["ROC_AUC"] < 0.7
