"""FastAPI inference service.

    python run.py api            (= uvicorn src.api.main:app --port 8000)

GET  /health                               status, data label (synthetic or real), model
GET  /forecast?date=YYYY-MM-DD&horizon=1   risk per module + top-3 modules
GET  /explain?module=FI&date=...&horizon=1 most influential neighbouring modules
GET  /history?module=FI&horizon=1          forecast vs actual over the test period
POST /recommend                            similar past incidents + their fixes
GET  /recommend/module?module=FI&date=...  fixes for a module flagged by the forecast

Models and the recommender index are loaded on first use (run `python run.py train` and
`python run.py index` first). Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.service import ForecastService, ServiceError
from src.utils import load_config

app = FastAPI(
    title="SAP incident forecasting API",
    version="1.0",
    description="Forecasts and resolution suggestions. Check /health for whether "
    "the data is SYNTHETIC.",
)


@lru_cache(maxsize=1)
def config() -> dict:
    return load_config()


@lru_cache(maxsize=1)
def forecaster() -> ForecastService:
    return ForecastService(config())


@lru_cache(maxsize=1)
def recommender():
    from src.recommender.recommend import Recommender

    return Recommender(config())


def _call(fn, *args, **kwargs):
    """Run a service call; translate errors into clear HTTP responses."""
    try:
        return fn(*args, **kwargs)
    except ServiceError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"not ready: {e}") from e


# ----------------------------------------------------------------------------- schemas
class ModuleForecast(BaseModel):
    module: str
    expected_incidents: float
    p1p2_risk: float
    actual_incidents: int | None = None
    actual_p1p2: bool | None = None


class ForecastResponse(BaseModel):
    date: str
    horizon: int
    origin_date: str
    model: str
    risk_score: str
    risk_note: str
    modules: list[ModuleForecast]
    top3: list[str]


class RecommendRequest(BaseModel):
    description: str = Field(
        min_length=3, max_length=500, examples=["Billing documents not posting to FI"]
    )
    module: str = Field(default="", examples=["SD"])
    k: int = Field(default=5, ge=1, le=20)
    same_module: bool = False


class Incident(BaseModel):
    incident_id: str
    module: str
    category: str
    priority: str
    short_description: str
    resolution_notes: str
    created_at: str
    similarity: float


class RecommendResponse(BaseModel):
    query: str
    module: str
    embedder: str
    results: list[Incident]


# ----------------------------------------------------------------------------- endpoints
@app.get("/health")
def health() -> dict:
    cfg = config()
    out = {
        "status": "ok",
        "model": cfg["api"]["model"],
        "recommender_embedder": cfg["recommender"]["embedder"],
    }
    try:
        f = forecaster()
        out.update(
            data="SYNTHETIC DATA" if f.daily.synthetic else "REAL DATA",
            synthetic=f.daily.synthetic,
            modules=f.modules,
            forecast_dates=f.date_range(),
        )
    except FileNotFoundError as e:
        out.update(status="degraded", detail=str(e))
    return out


@app.get("/forecast", response_model=ForecastResponse)
def forecast(date: str = Query(examples=["2025-11-03"]), horizon: int = Query(1, ge=1, le=7)):
    return _call(forecaster().forecast, date, horizon)


@app.get("/explain")
def explain(module: str, date: str, horizon: int = Query(1, ge=1, le=7)) -> dict:
    return _call(forecaster().explain, module, date, horizon)


@app.get("/history")
def history(module: str, horizon: int = Query(1, ge=1, le=7)) -> dict:
    return _call(forecaster().history, module, horizon)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if req.module and req.module not in config()["data"]["modules"]:
        raise HTTPException(422, f"unknown module {req.module!r}")
    rec = _call(recommender)
    hits = rec.similar(req.description, req.module, req.k, req.same_module)
    return {"query": req.description, "module": req.module, "embedder": rec.name, "results": hits}


@app.get("/recommend/module")
def recommend_module(module: str, date: str | None = None) -> dict:
    if module not in config()["data"]["modules"]:
        raise HTTPException(422, f"unknown module {module!r}")
    return _call(lambda: _call(recommender).for_module(module, before=date))
