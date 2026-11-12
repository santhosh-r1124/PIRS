"""Data contract: the format every data source (synthetic or real) must be converted to.

Three tables:
  incidents       one row per ticket (INCIDENT_COLUMNS)
  daily signals   one row per (date, module): job_failures, idoc_errors, change_events
  calendar        one row per date: is_weekend, is_month_end_close, is_quarter_end

Validators never fix data; they return a ValidationReport listing errors (data must not
be used), warnings (worth a look) and missing-value counts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PRIORITIES = ["P1", "P2", "P3", "P4"]

# column -> kind ("text" or "datetime")
INCIDENT_COLUMNS = {
    "incident_id": "text",
    "created_at": "datetime",
    "module": "text",
    "priority": "text",
    "category": "text",
    "short_description": "text",
    "resolution_notes": "text",  # used by the resolution recommender
    "resolved_at": "datetime",  # optional: open tickets have no value
}
INCIDENT_REQUIRED = ["incident_id", "created_at", "module", "priority", "category"]

SIGNAL_COLUMNS = ["date", "module", "job_failures", "idoc_errors", "change_events"]
CALENDAR_COLUMNS = ["date", "is_weekend", "is_month_end_close", "is_quarter_end"]


@dataclass
class ValidationReport:
    table: str
    n_rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self) -> str:
        lines = [f"[{self.table}] {self.n_rows} rows -> {'OK' if self.ok else 'INVALID'}"]
        lines += [f"  ERROR   {e}" for e in self.errors]
        lines += [f"  warning {w}" for w in self.warnings]
        gaps = {c: n for c, n in self.missing.items() if n}
        lines.append(f"  missing values: {gaps if gaps else 'none'}")
        return "\n".join(lines)

    def raise_if_errors(self) -> None:
        if self.errors:
            raise ValueError(str(self))


# --------------------------------------------------------------------------- helpers
def _is_text(s: pd.Series) -> bool:
    return bool(s.dropna().map(lambda v: isinstance(v, str)).all())


def _check_datetime(df: pd.DataFrame, col: str, now: pd.Timestamp, rep: ValidationReport):
    s = df[col]
    if not pd.api.types.is_datetime64_any_dtype(s):
        rep.errors.append(f"{col}: expected datetime, got {s.dtype}")
        return
    if getattr(s.dt, "tz", None) is not None:
        rep.errors.append(f"{col}: timezone-aware; convert to naive local time first")
        return
    n_future = int((s > now).sum())
    if n_future:
        rep.errors.append(f"{col}: {n_future} value(s) in the future (after {now})")


def _check_allowed(s: pd.Series, allowed: Iterable[str], label: str) -> str | None:
    bad = sorted(set(s.dropna()) - set(allowed))
    if bad:
        shown = bad[:5] + (["..."] if len(bad) > 5 else [])
        return f"{label}: {len(bad)} unknown value(s): {shown}"
    return None


# --------------------------------------------------------------------------- incidents
def validate_incidents(
    df: pd.DataFrame,
    modules: Iterable[str],
    categories: Iterable[str] | None = None,
    priorities: Iterable[str] = PRIORITIES,
    now: pd.Timestamp | None = None,
) -> ValidationReport:
    now = now if now is not None else pd.Timestamp.now()
    rep = ValidationReport("incidents", len(df))

    missing_cols = [c for c in INCIDENT_COLUMNS if c not in df.columns]
    if missing_cols:
        rep.errors.append(f"missing columns: {missing_cols}")
        return rep
    extra = [c for c in df.columns if c not in INCIDENT_COLUMNS]
    if extra:
        rep.warnings.append(f"extra columns are ignored downstream: {extra}")

    rep.missing = {c: int(df[c].isna().sum()) for c in INCIDENT_COLUMNS}
    for c in INCIDENT_REQUIRED:
        if rep.missing[c]:
            rep.errors.append(f"{c}: {rep.missing[c]} missing value(s) in a required column")
    for c in ["short_description", "resolution_notes"]:
        if rep.missing[c]:
            rep.warnings.append(f"{c}: {rep.missing[c]} missing value(s)")

    for col, kind in INCIDENT_COLUMNS.items():
        if kind == "text" and not _is_text(df[col]):
            rep.errors.append(f"{col}: expected text values")
        elif kind == "datetime":
            _check_datetime(df, col, now, rep)

    n_dup = int(df["incident_id"].duplicated().sum())
    if n_dup:
        rep.errors.append(f"incident_id: {n_dup} duplicate(s)")

    for msg in [
        _check_allowed(df["module"], modules, "module"),
        _check_allowed(df["priority"], priorities, "priority"),
    ]:
        if msg:
            rep.errors.append(msg)
    if categories is not None:
        msg = _check_allowed(df["category"], categories, "category")
        if msg:
            rep.warnings.append(msg)

    # Only compare the two timestamps if both passed the datetime checks above.
    dt_ok = all(
        pd.api.types.is_datetime64_any_dtype(df[c]) and getattr(df[c].dt, "tz", None) is None
        for c in ["created_at", "resolved_at"]
    )
    if dt_ok:
        n_neg = int((df["resolved_at"] < df["created_at"]).sum())
        if n_neg:
            rep.errors.append(f"resolved_at before created_at in {n_neg} row(s)")
    return rep


# --------------------------------------------------------------------------- daily signals
def validate_daily_signals(
    df: pd.DataFrame, modules: Iterable[str], now: pd.Timestamp | None = None
) -> ValidationReport:
    now = now if now is not None else pd.Timestamp.now()
    modules = list(modules)
    rep = ValidationReport("daily_signals", len(df))

    missing_cols = [c for c in SIGNAL_COLUMNS if c not in df.columns]
    if missing_cols:
        rep.errors.append(f"missing columns: {missing_cols}")
        return rep

    rep.missing = {c: int(df[c].isna().sum()) for c in SIGNAL_COLUMNS}
    for c, n in rep.missing.items():
        if n:
            rep.errors.append(f"{c}: {n} missing value(s)")

    _check_datetime(df, "date", now, rep)
    msg = _check_allowed(df["module"], modules, "module")
    if msg:
        rep.errors.append(msg)

    for c in ["job_failures", "idoc_errors", "change_events"]:
        s = df[c].dropna()
        if not pd.api.types.is_numeric_dtype(s):
            rep.errors.append(f"{c}: expected numbers, got {s.dtype}")
        elif (s < 0).any() or not np.allclose(s, np.round(s)):
            rep.errors.append(f"{c}: must be non-negative whole numbers")

    n_dup = int(df.duplicated(["date", "module"]).sum())
    if n_dup:
        rep.errors.append(f"(date, module): {n_dup} duplicate row(s)")
    elif pd.api.types.is_datetime64_any_dtype(df["date"]) and len(df):
        n_days = df["date"].dt.normalize().nunique()
        if len(df) != n_days * len(modules):
            rep.warnings.append(
                f"incomplete grid: {len(df)} rows for {n_days} days x {len(modules)} modules"
            )
    return rep


# --------------------------------------------------------------------------- calendar
def build_calendar(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    close_last_working_days: int = 3,
    close_first_working_days: int = 2,
    quarter_end_months: Iterable[int] = (3, 6, 9, 12),
) -> pd.DataFrame:
    """Daily calendar flags.

    is_month_end_close: last `close_last_working_days` working days of a month, plus the
        first `close_first_working_days` working days of the next month (books for the
        previous month are still being closed then).
    is_quarter_end: a close day whose close belongs to a quarter-end month (Mar/Jun/Sep/Dec).
    Working day = Monday-Friday.
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    # Compute on whole months so "last N working days" is right at the range edges.
    full = pd.date_range(start.replace(day=1), end + pd.offsets.MonthEnd(0), freq="D")
    cal = pd.DataFrame({"date": full})
    cal["is_weekend"] = cal["date"].dt.dayofweek >= 5
    working = (~cal["is_weekend"]).astype(int)
    month = cal["date"].dt.to_period("M")

    wd_from_start = working.groupby(month).cumsum()
    wd_from_end = working[::-1].groupby(month[::-1]).cumsum()[::-1]
    is_working = working.astype(bool)
    in_last = is_working & (wd_from_end <= close_last_working_days)
    in_first = is_working & (wd_from_start <= close_first_working_days)

    cal["is_month_end_close"] = in_last | in_first
    # month whose books are being closed: same month for "last", previous month for "first"
    close_month = np.where(in_first, (month - 1).dt.month, month.dt.month)
    cal["is_quarter_end"] = cal["is_month_end_close"] & np.isin(
        close_month, list(quarter_end_months)
    )
    cal = cal[(cal["date"] >= start) & (cal["date"] <= end)].reset_index(drop=True)
    return cal[CALENDAR_COLUMNS]


def validate_calendar(df: pd.DataFrame) -> ValidationReport:
    rep = ValidationReport("calendar", len(df))
    missing_cols = [c for c in CALENDAR_COLUMNS if c not in df.columns]
    if missing_cols:
        rep.errors.append(f"missing columns: {missing_cols}")
        return rep
    rep.missing = {c: int(df[c].isna().sum()) for c in CALENDAR_COLUMNS}
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        rep.errors.append(f"date: expected datetime, got {df['date'].dtype}")
        return rep
    if df["date"].duplicated().any():
        rep.errors.append("date: duplicates")
    gaps = df["date"].sort_values().diff().dropna()
    if (gaps != pd.Timedelta(days=1)).any():
        rep.errors.append("date: not a continuous daily range")
    for c in CALENDAR_COLUMNS[1:]:
        if not pd.api.types.is_bool_dtype(df[c]):
            rep.errors.append(f"{c}: expected boolean, got {df[c].dtype}")
    if (df["is_quarter_end"] & ~df["is_month_end_close"]).any():
        rep.errors.append("is_quarter_end set on a day that is not a close day")
    return rep
