"""Evaluation helpers: per-target rank correlation, paired ablation differences."""

import numpy as np
import pytest

from src.evaluate import ablations, per_target_spearman


def test_per_target_spearman_ignores_per_target_scale():
    rng = np.random.default_rng(0)
    w = rng.random((5, 5))
    np.fill_diagonal(w, 0)
    sources = {j: [i for i in range(5) if i != j] for j in range(5)}
    # same ranking within each target, but every target column on its own scale
    rescaled = w * np.arange(1, 6)[None, :]
    assert per_target_spearman(rescaled, w, sources) == pytest.approx(1.0)
    assert per_target_spearman(-w, w, sources) == pytest.approx(-1.0)
    assert np.isnan(per_target_spearman(w, w, {0: [1, 2]}))  # needs >= 3 candidate sources


def test_paired_ablation_difference():
    def run(model, seed, mae):
        return {
            "model": model,
            "target": "regression",
            "seed": seed,
            "split": "main",
            "metrics": {"test": {"h1": {"MAE": mae, "PoissonDev": 1.0, "P@3": 0.5}}},
        }

    runs = [
        run("gat_transformer", 1, 0.70),
        run("gat_transformer", 2, 0.72),
        run("gat_transformer_none", 1, 0.69),
        run("gat_transformer_none", 2, 0.71),
    ]
    _, diff = ablations(runs)
    row = diff.set_index("variant").loc["gat_transformer_none"]
    assert row["MAE@1 (counts)"] == "-0.010 ± 0.000"  # same seed, variant - main model
