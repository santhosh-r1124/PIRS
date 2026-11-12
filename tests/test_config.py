"""The config is the single source of truth, so catch broken edits early."""

import pytest

from src.utils import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_required_sections(cfg):
    for section in ["project", "paths", "data", "graph", "dataset"]:
        assert section in cfg, f"missing section: {section}"


def test_modules_unique(cfg):
    modules = cfg["data"]["modules"]
    assert len(modules) == len(set(modules)) >= 2


def test_edges_are_valid(cfg):
    modules = set(cfg["data"]["modules"])
    edges = [tuple(e) for e in cfg["graph"]["edges"]]
    assert all(len(e) == 2 for e in edges)
    assert all(a in modules and b in modules for a, b in edges), "edge uses unknown module"
    assert all(a != b for a, b in edges), "self-loop in edge list"
    assert len(edges) == len(set(edges)), "duplicate edge"


def test_basis_feeds_every_module(cfg):
    """BASIS is the technical platform, so it must have an edge into every other module."""
    targets = {b for a, b in cfg["graph"]["edges"] if a == "BASIS"}
    assert targets == set(cfg["data"]["modules"]) - {"BASIS"}


def test_split_is_chronological_partition(cfg):
    split = cfg["dataset"]["split"]
    assert list(split) == ["train", "val", "test"], "split order must be train, val, test"
    assert abs(sum(split.values()) - 1.0) < 1e-9
    assert all(v > 0 for v in split.values())


def test_window_settings(cfg):
    ds = cfg["dataset"]
    assert ds["lookback"] > 0
    assert all(h > 0 for h in ds["horizons"])
    assert ds["target"] in {"regression", "classification"}


def test_deep_merge_overrides_only_listed_keys():
    from src.utils import deep_merge

    base = {"a": {"x": 1, "y": [1, 2]}, "b": 2}
    out = deep_merge(base, {"a": {"y": [3]}, "c": 4})
    assert out == {"a": {"x": 1, "y": [3]}, "b": 2, "c": 4}
    assert base["a"]["y"] == [1, 2]  # input not modified


def test_quick_config_inherits_and_isolates(cfg):
    quick = load_config("configs/quick.yaml")
    assert "base" not in quick
    assert quick["synthetic"]["n_days"] < cfg["synthetic"]["n_days"]
    assert quick["training"]["max_epochs"] < cfg["training"]["max_epochs"]
    assert quick["graph"] == cfg["graph"] and quick["dataset"] == cfg["dataset"]  # inherited
    assert quick["synthetic"]["propagation"] == cfg["synthetic"]["propagation"]
    # every output goes to separate folders, never over the real data/ or results/
    for key in ["synthetic", "processed", "results", "figures", "checkpoints"]:
        assert quick["paths"][key] != cfg["paths"][key]
        assert quick["paths"][key].split("/")[0] in {"data_quick", "results_quick"}
    assert quick["recommender"]["chroma_dir"].startswith("results_quick/")


def test_sap_config_env_var(monkeypatch):
    monkeypatch.setenv("SAP_CONFIG", "configs/quick.yaml")
    assert load_config()["paths"]["results"] == "results_quick"
    monkeypatch.delenv("SAP_CONFIG")
    assert load_config()["paths"]["results"] == "results"
