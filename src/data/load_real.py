"""Adapter: real ticket export (CSV from ServiceNow, SAP Solution Manager, ...) -> schema.

Everything export-specific lives in `real_data` in the config:
  column_map   schema column -> column name in the export
  value_maps   export value -> schema value (priority labels, assignment groups -> modules, ...)
If no export is present the adapter says so and returns None, so the rest of the project
runs on synthetic data.

    python -m src.data.load_real
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schema import (
    INCIDENT_COLUMNS,
    INCIDENT_REQUIRED,
    SIGNAL_COLUMNS,
    validate_daily_signals,
    validate_incidents,
)
from src.utils import ROOT, load_config


def _to_naive_datetime(s: pd.Series, fmt: str | None) -> pd.Series:
    out = pd.to_datetime(s, format=fmt, errors="coerce")
    if getattr(out.dt, "tz", None) is not None:  # e.g. "2024-03-01T10:00:00Z"
        out = out.dt.tz_convert(None)
    return out


def map_incidents(raw: pd.DataFrame, rcfg: dict, modules: list[str]) -> tuple[pd.DataFrame, dict]:
    """Rename, recode and type-convert an export. Returns (schema DataFrame, statistics)."""
    cmap = rcfg["column_map"]
    absent = {col: src for col, src in cmap.items() if src not in raw.columns}
    missing_required = [f"{col} (export column '{src}')" for col, src in absent.items()
                        if col in INCIDENT_REQUIRED]  # fmt: skip
    if missing_required:
        raise ValueError(f"export lacks required columns: {missing_required}")

    df = pd.DataFrame(index=raw.index)
    for col in INCIDENT_COLUMNS:
        src = cmap.get(col)
        df[col] = raw[src] if src in raw.columns else pd.NA

    for col, mapping in (rcfg.get("value_maps") or {}).items():
        if mapping:
            df[col] = df[col].map(lambda v, m=mapping: m.get(v, v))

    stats = {"rows_in": len(raw), "optional_columns_absent": sorted(absent)}
    for col, kind in INCIDENT_COLUMNS.items():
        if kind == "datetime":
            before = df[col].notna().sum()
            df[col] = _to_naive_datetime(df[col], rcfg.get("datetime_format"))
            stats[f"unparseable_{col}"] = int(before - df[col].notna().sum())
        else:
            df[col] = df[col].map(lambda v: str(v).strip() if pd.notna(v) else None)
            df[col] = df[col].astype(object)

    unknown = ~df["module"].isin(modules)
    stats["rows_unknown_module"] = int(unknown.sum())
    if rcfg.get("drop_unknown_modules", True):
        df = df[~unknown].reset_index(drop=True)
    stats["rows_out"] = len(df)
    return df, stats


def load_real(cfg: dict, path: str | Path | None = None) -> pd.DataFrame | None:
    """Load and validate a real incident export, or return None if there is none."""
    rcfg = cfg["real_data"]
    path = Path(path) if path else ROOT / rcfg["incidents_csv"]
    if not path.exists():
        print(f"No real incident export at {path} - skipping.")
        return None
    raw = pd.read_csv(path, dtype=str, keep_default_na=True)  # keep IDs/labels as text
    df, stats = map_incidents(raw, rcfg, list(cfg["data"]["modules"]))
    print(f"real incidents: {stats}")
    rep = validate_incidents(df, cfg["data"]["modules"], cfg["data"]["categories"],
                             cfg["data"]["priorities"])  # fmt: skip
    print(rep)
    rep.raise_if_errors()
    return df


def load_real_signals(cfg: dict, path: str | Path | None = None) -> pd.DataFrame | None:
    """Optional daily signals export (date, module, job_failures, idoc_errors, change_events)."""
    path = Path(path) if path else ROOT / cfg["real_data"]["signals_csv"]
    if not path.exists():
        print(f"No real daily-signals export at {path} - skipping.")
        return None
    df = pd.read_csv(path)
    df["date"] = _to_naive_datetime(df["date"], None).dt.normalize()
    rep = validate_daily_signals(df[SIGNAL_COLUMNS], cfg["data"]["modules"])
    print(rep)
    rep.raise_if_errors()
    return df[SIGNAL_COLUMNS]


def main() -> None:
    cfg = load_config()
    load_real(cfg)
    load_real_signals(cfg)


if __name__ == "__main__":
    main()
