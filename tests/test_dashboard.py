"""Dashboard runs end to end: Streamlit's AppTest executes the script, and its HTTP calls are
routed to the real API in-process (no server, no browser)."""

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("torch")

import requests  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.api import main  # noqa: E402
from src.utils import ROOT  # noqa: E402
from tests.test_api import CFG, needs_ckpt  # noqa: E402

APP = str(ROOT / "src" / "dashboard" / "app.py")


@pytest.fixture(scope="module")
def routed(data_ready, tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("CHROMA_DIR", str(tmp_path_factory.mktemp("chroma")))
    from src.recommender.build_index import build

    build(CFG, "tfidf", verbose=False)
    for f in [main.forecaster, main.recommender, main.config]:
        f.cache_clear()
    client = TestClient(main.app)
    mp.setattr(requests, "request", lambda method, url, **kw: client.request(method, url, **kw))
    yield
    mp.undo()


@needs_ckpt
def test_forecast_page(routed):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert any("SYNTHETIC DATA" in w.value for w in at.warning)
    assert any("Forecast for" in s.value for s in at.subheader)
    assert len(at.dataframe) >= 2  # risk table + similar incidents


@needs_ckpt
def test_comparison_page(routed):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value("Model comparison").run()
    assert not at.exception, at.exception
    assert any("Model comparison" in s.value for s in at.subheader)


def test_api_down_is_reported(monkeypatch):
    def refuse(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "request", refuse)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert any("not reachable" in e.value for e in at.error)
