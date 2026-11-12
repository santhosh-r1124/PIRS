"""Model serving logic behind the API (kept separate from FastAPI so it is easy to test).

A forecast for target day D at horizon h is made at the end of the origin day t = D - h,
from the 28 days up to t - exactly how the models were trained and evaluated. Dates after
the last day of data are allowed up to t = last day (the calendar of future days is known
in advance); actual values are returned when D is inside the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.data.build_dataset import (
    DailyData,
    Scaler,
    chronological_split,
    load_daily,
)
from src.data.schema import build_calendar
from src.graph.sap_graph import build_graph
from src.models.trainer import Objective
from src.utils import get_path


class ServiceError(ValueError):
    """Bad request (unknown module, date out of range, ...)."""


def calendar_features(dates: pd.DatetimeIndex, cfg: dict) -> np.ndarray:
    """Same five calendar features as build_dataset, for any dates (also future ones)."""
    c = cfg["calendar"]
    cal = build_calendar(
        dates.min(),
        dates.max(),
        c["close_last_working_days"],
        c["close_first_working_days"],
        c["quarter_end_months"],
    )
    cal = cal.set_index("date").loc[dates]
    dow = dates.dayofweek.to_numpy()
    return np.column_stack(
        [
            cal["is_weekend"].to_numpy(float),
            cal["is_month_end_close"].to_numpy(float),
            cal["is_quarter_end"].to_numpy(float),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
        ]
    ).astype(np.float32)


@dataclass
class LoadedModel:
    model: torch.nn.Module
    objective: Objective
    scaled: np.ndarray  # [T, N, F] features scaled with the checkpoint's training scaler


class ForecastService:
    def __init__(self, cfg: dict):
        from src.train import build_model, run_name

        self.cfg, self.daily = cfg, load_daily(cfg)
        d: DailyData = self.daily
        ds = cfg["dataset"]
        self.lookback, self.horizon = ds["lookback"], max(ds["horizons"])
        self.model_name, self.seed = cfg["api"]["model"], cfg["api"]["seed"]
        self.graph = build_graph(cfg)
        last = d.dates[-1]
        future = pd.date_range(last + pd.Timedelta(days=1), periods=self.horizon)
        self.calendar = np.concatenate([d.future_calendar, calendar_features(future, cfg)])
        self.all_dates = d.dates.append(future)
        self.split = chronological_split(len(d.dates), self.lookback, self.horizon, ds["split"])
        self.models = {}
        for target in ["regression", "classification"]:
            path = (
                get_path(cfg, "checkpoints") / f"{run_name(self.model_name, target, self.seed)}.pt"
            )
            if not path.exists():
                raise FileNotFoundError(f"missing checkpoint {path}: run `python run.py train`")
            ck = torch.load(path, weights_only=False)  # our own file, written by src.train
            feats = ck["feature_names"]
            shape = {
                "train": {
                    "X": np.zeros((1, self.lookback, len(d.modules), len(feats))),
                    "Xf": np.zeros((1, self.horizon, self.calendar.shape[1])),
                },
                "lookback": self.lookback,
                "horizon": self.horizon,
            }
            model = build_model(self.model_name, cfg, d, shape)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            y_train = d.target(target)[: self.split.train_end]
            scaled = Scaler(**ck["scaler"]).transform(d.features)[:, :, d.feature_index(feats)]
            self.models[target] = LoadedModel(model, Objective(target, y_train), scaled)

    # ------------------------------------------------------------------ helpers
    @property
    def modules(self) -> list[str]:
        return self.daily.modules

    def check_module(self, module: str) -> int:
        if module not in self.modules:
            raise ServiceError(f"unknown module {module!r}; choose from {self.modules}")
        return self.modules.index(module)

    def date_range(self) -> tuple[str, str]:
        first = self.daily.dates[self.lookback - 1 + 1]
        last = self.daily.dates[-1] + pd.Timedelta(days=1)
        return str(first.date()), str(last.date())

    def origin(self, date: str, horizon: int) -> int:
        if not 1 <= horizon <= self.horizon:
            raise ServiceError(f"horizon must be between 1 and {self.horizon}")
        try:
            target = pd.Timestamp(date).normalize()
        except ValueError as e:
            raise ServiceError(f"invalid date {date!r}; use YYYY-MM-DD") from e
        t = int((target - self.daily.dates[0]).days) - horizon
        if t < self.lookback - 1 or t > len(self.daily.dates) - 1:
            first = self.daily.dates[self.lookback - 1] + pd.Timedelta(days=horizon)
            last = self.daily.dates[-1] + pd.Timedelta(days=horizon)
            raise ServiceError(
                f"for horizon {horizon} the date must be between {first.date()} and {last.date()}"
            )
        return t

    def _window(self, target: str, origins: list[int]) -> dict:
        lm = self.models[target]
        past = np.asarray(origins)[:, None] + np.arange(-self.lookback + 1, 1)[None, :]
        fut = np.asarray(origins)[:, None] + np.arange(1, self.horizon + 1)[None, :]
        return {"X": torch.as_tensor(lm.scaled[past]), "Xf": torch.as_tensor(self.calendar[fut])}

    @torch.no_grad()
    def _predict(self, target: str, origins: list[int], keep_attention: bool = False):
        lm, w = self.models[target], self._window(target, origins)
        kwargs = {"keep_attention": True} if keep_attention else {}
        return lm.objective.predict(lm.model(w["X"], w["Xf"], **kwargs)).numpy(), lm.model

    # ------------------------------------------------------------------ endpoints
    def forecast(self, date: str, horizon: int = 1) -> dict:
        t = self.origin(date, horizon)
        counts = self._predict("regression", [t])[0][0, horizon - 1]
        risk = self._predict("classification", [t])[0][0, horizon - 1]
        day = t + horizon
        known = day < len(self.daily.dates)
        rows = []
        for i, m in enumerate(self.modules):
            row = {
                "module": m,
                "expected_incidents": round(float(counts[i]), 3),
                "p1p2_risk": round(float(risk[i]), 3),
            }
            if known:
                row["actual_incidents"] = int(self.daily.counts[day, i])
                row["actual_p1p2"] = bool(self.daily.p1p2[day, i] > 0)
            rows.append(row)
        top = sorted(rows, key=lambda r: -r["p1p2_risk"])[:3]
        return {
            "date": str(self.all_dates[day].date()),
            "horizon": horizon,
            "origin_date": str(self.daily.dates[t].date()),
            "model": self.model_name,
            "risk_score": "p1p2_risk",
            "risk_note": "p1p2_risk ranks modules by the chance of a P1/P2 incident; the model "
            "was trained with positive-class weighting, so it is a ranking score, not a "
            "calibrated probability.",
            "modules": rows,
            "top3": [r["module"] for r in top],
        }

    def explain(self, module: str, date: str, horizon: int = 1) -> dict:
        j = self.check_module(module)
        t = self.origin(date, horizon)
        _, model = self._predict("classification", [t], keep_attention=True)
        out = {
            "module": module,
            "date": str(self.all_dates[t + horizon].date()),
            "model": self.model_name,
            "note": "Attention shows which modules' recent data the model weighted; it is "
            "not proof of cause. On the synthetic data it did not match the true "
            "propagation (results/summary.md, section 5).",
        }
        att = getattr(model, "last_attention", None)
        if att is not None:
            src, dst = model.edge_index.tolist()
            w = att["edges"].mean(-1).tolist()
            infl = [
                {"source": self.modules[s], "weight": round(x, 3)}
                for s, d_, x in zip(src, dst, w, strict=True)
                if d_ == j
            ]
            out["influences"] = sorted(infl, key=lambda r: -r["weight"])
            out["self_weight"] = round(float(att["self"][j].mean()), 3)
        adj = model.learned_adjacency() if hasattr(model, "learned_adjacency") else None
        if adj is not None:
            col = adj[:, j].tolist()
            out["learned_adjacency"] = sorted(
                [
                    {"source": m, "weight": round(c, 3)}
                    for m, c in zip(self.modules, col, strict=True)
                    if m != module
                ],
                key=lambda r: -r["weight"],
            )[:5]
        return out

    def history(self, module: str, horizon: int = 1) -> dict:
        """Forecast vs actual over the TEST period (data the model never trained on)."""
        j = self.check_module(module)
        if not 1 <= horizon <= self.horizon:
            raise ServiceError(f"horizon must be between 1 and {self.horizon}")
        origins = self.split.test.tolist()
        pred = self._predict("regression", origins)[0][:, horizon - 1, j]
        days = np.asarray(origins) + horizon
        return {
            "module": module,
            "horizon": horizon,
            "period": "test",
            "model": self.model_name,
            "dates": [str(self.daily.dates[d].date()) for d in days],
            "expected": [round(float(x), 3) for x in pred],
            "actual": [int(self.daily.counts[d, j]) for d in days],
        }
