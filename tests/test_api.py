"""API endpoints via FastAPI's TestClient.

Validation tests always run. Tests that need trained checkpoints are skipped (with the
reason) until `python run.py train` has produced them; the recommender tests build a small
TF-IDF index in a temporary folder.
"""

import pytest

pytest.importorskip("fastapi")
torch = pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

from src.api import main  # noqa: E402
from src.train import run_name  # noqa: E402
from src.utils import get_path, load_config  # noqa: E402

CFG = load_config()
HAVE_CKPT = all(
    (
        get_path(CFG, "checkpoints") / f"{run_name(CFG['api']['model'], t, CFG['api']['seed'])}.pt"
    ).exists()
    for t in ["regression", "classification"]
)
needs_ckpt = pytest.mark.skipif(not HAVE_CKPT, reason="run `python run.py train` first")


@pytest.fixture(scope="module")
def client(data_ready, tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("CHROMA_DIR", str(tmp_path_factory.mktemp("chroma")))
    from src.recommender.build_index import build

    build(CFG, "tfidf", verbose=False)
    for f in [main.forecaster, main.recommender, main.config]:
        f.cache_clear()
    yield TestClient(main.app)
    mp.undo()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    if body["status"] == "ok":
        assert body["synthetic"] is True and body["data"] == "SYNTHETIC DATA"


@needs_ckpt
def test_forecast(client):
    r = client.get("/forecast", params={"date": "2025-11-03", "horizon": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["origin_date"] == "2025-11-02" and len(body["modules"]) == 10
    assert len(body["top3"]) == 3
    by_risk = sorted(body["modules"], key=lambda m: -m["p1p2_risk"])
    assert body["top3"] == [m["module"] for m in by_risk[:3]]
    for m in body["modules"]:
        assert m["expected_incidents"] >= 0 and 0 <= m["p1p2_risk"] <= 1
        assert m["actual_incidents"] is not None  # date inside the data -> actuals known


@needs_ckpt
def test_forecast_beyond_the_data(client):
    r = client.get("/forecast", params={"date": "2026-01-03", "horizon": 3})
    assert r.status_code == 200, r.text
    assert all(m["actual_incidents"] is None for m in r.json()["modules"])


@needs_ckpt
@pytest.mark.parametrize(
    "params, status",
    [
        ({"date": "2021-01-01"}, 422),
        ({"date": "not-a-date"}, 422),
        ({"date": "2025-11-03", "horizon": 9}, 422),
        ({}, 422),
    ],
)
def test_forecast_rejects_bad_input(client, params, status):
    r = client.get("/forecast", params=params)
    assert r.status_code == status
    assert "detail" in r.json()


@needs_ckpt
def test_explain(client):
    r = client.get("/explain", params={"module": "FI", "date": "2025-11-03"})
    assert r.status_code == 200, r.text
    body = r.json()
    sources = {i["source"] for i in body["influences"]}
    assert sources == {"SD", "MM", "HCM", "BASIS"}  # FI's incoming edges in the graph
    total = sum(i["weight"] for i in body["influences"]) + body["self_weight"]
    assert total == pytest.approx(1.0, abs=0.01)
    assert "not proof" in body["note"]
    assert client.get("/explain", params={"module": "XX", "date": "2025-11-03"}).status_code == 422


@needs_ckpt
def test_history_is_test_period(client):
    body = client.get("/history", params={"module": "SD"}).json()
    assert body["period"] == "test" and body["dates"][0] >= "2025-07-21"
    assert len(body["dates"]) == len(body["expected"]) == len(body["actual"])


def test_recommend(client):
    r = client.post("/recommend", json={"description": "IDocs failing for INVOIC", "module": "SD"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["embedder"] == CFG["recommender"]["embedder"] and len(body["results"]) == 5
    assert body["results"][0]["resolution_notes"]


@pytest.mark.parametrize(
    "payload",
    [
        {"description": "x"},
        {"description": "valid text", "module": "XX"},
        {"description": "valid text", "k": 0},
        {},
    ],
)
def test_recommend_rejects_bad_input(client, payload):
    assert client.post("/recommend", json=payload).status_code == 422


def test_recommend_for_module(client):
    r = client.get("/recommend/module", params={"module": "MM", "date": "2025-01-31"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["typical_category"] and {i["module"] for i in body["incidents"]} == {"MM"}
