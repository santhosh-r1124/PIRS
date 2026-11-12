"""Recommender: index holds only past tickets, retrieval works, baselines, unavailable models."""

import pandas as pd
import pytest

pytest.importorskip("chromadb")

from src.recommender.build_index import build, incident_periods  # noqa: E402
from src.recommender.embedders import EmbedderUnavailable, get_embedder  # noqa: E402
from src.recommender.recommend import Recommender, random_baselines  # noqa: E402
from src.utils import load_config  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def rec(cfg, data_ready, tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("CHROMA_DIR", str(tmp_path_factory.mktemp("chroma")))
    build(cfg, "tfidf", verbose=False)
    yield Recommender(cfg, "tfidf")
    mp.undo()


def test_index_contains_only_training_period(cfg, rec):
    train, test, period = incident_periods(cfg)
    assert rec.col.count() == len(train)
    assert train["created_at"].max() < pd.Timestamp(period["index_until"])
    assert test["created_at"].min() >= pd.Timestamp(period["test_from"])
    got = rec.col.get(ids=test["incident_id"].head(20).tolist())
    assert got["ids"] == []  # no held-out ticket is in the index


def test_similar_returns_ranked_past_fixes(rec):
    hits = rec.similar("Background job SD_BILLING_DUE_LIST cancelled", module="SD")
    assert len(hits) == 5
    assert {"incident_id", "module", "category", "resolution_notes", "similarity"} <= set(hits[0])
    sims = [h["similarity"] for h in hits]
    assert sims == sorted(sims, reverse=True)
    assert hits[0]["category"] == "job_failure"
    same = rec.similar("Background job cancelled", module="FI", same_module=True)
    assert {h["module"] for h in same} == {"FI"}


def test_for_flagged_module(rec):
    out = rec.for_module("MM", before="2025-01-31")
    assert out["typical_category"] is not None
    assert out["incidents"] and {h["module"] for h in out["incidents"]} == {"MM"}
    assert {h["category"] for h in out["incidents"]} == {out["typical_category"]}


def test_random_baselines_by_hand():
    train = pd.DataFrame({"module": ["A", "A", "A", "B"], "category": ["x", "x", "y", "y"]})
    test = pd.DataFrame({"module": ["A"], "category": ["x"]})
    b = random_baselines(train, test, k=2)
    assert b["random_any_ticket"]["hit@1"] == pytest.approx(0.5)
    assert b["random_same_module"]["hit@1"] == pytest.approx(2 / 3)
    assert b["random_same_module"]["hit@2"] == pytest.approx(1 - (1 / 3) ** 2)


def test_missing_onnx_model_is_reported_not_faked(cfg, tmp_path, monkeypatch):
    rcfg = dict(cfg["recommender"], onnx_archive=str(tmp_path / "missing.tar.gz"))
    import chromadb.utils.embedding_functions as ef

    monkeypatch.setattr(ef.ONNXMiniLM_L6_V2, "DOWNLOAD_PATH", tmp_path / "cache")
    with pytest.raises(EmbedderUnavailable):
        get_embedder("minilm_onnx", rcfg)
