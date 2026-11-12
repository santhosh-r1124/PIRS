"""Data contract: validators catch bad data; the calendar follows the close definition."""

import pandas as pd
import pytest

from src.data.schema import (
    build_calendar,
    validate_calendar,
    validate_daily_signals,
    validate_incidents,
)

MODULES = ["FI", "SD", "BASIS"]
CATS = ["job_failure", "posting_error"]
NOW = pd.Timestamp("2026-01-01")


def incidents(**overrides):
    df = pd.DataFrame({
        "incident_id": ["INC1", "INC2", "INC3"],
        "created_at": pd.to_datetime(["2025-01-02 09:00", "2025-01-02 10:00", "2025-01-03 11:00"]),
        "module": ["FI", "SD", "BASIS"],
        "priority": ["P1", "P3", "P4"],
        "category": ["posting_error", "job_failure", "job_failure"],
        "short_description": ["Period closed", "Billing job failed", "Job cancelled"],
        "resolution_notes": ["Opened period", "Restarted job", "Restarted job"],
        "resolved_at": pd.to_datetime(["2025-01-02 12:00", None, "2025-01-04 08:00"]),
    })  # fmt: skip
    for col, val in overrides.items():
        df[col] = val
    return df


def check(df, **kw):
    return validate_incidents(df, MODULES, CATS, now=NOW, **kw)


def test_valid_incidents_pass():
    rep = check(incidents())
    assert rep.ok, rep
    assert rep.missing["resolved_at"] == 1  # optional column: reported, not an error


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"module": ["FI", "XX", "BASIS"]}, "module"),
        ({"priority": ["P1", "P9", "P4"]}, "priority"),
        ({"incident_id": ["INC1", "INC1", "INC3"]}, "duplicate"),
        ({"created_at": pd.to_datetime(["2025-01-02", "2030-01-01", "2025-01-03"])}, "future"),
        ({"created_at": ["2025-01-02", "2025-01-02", "2025-01-03"]}, "expected datetime"),
        ({"module": ["FI", None, "BASIS"]}, "missing value"),
        ({"resolved_at": pd.to_datetime(["2025-01-01", None, None])}, "before created_at"),
        ({"created_at": pd.to_datetime(["2025-01-02"] * 3).tz_localize("UTC")}, "timezone"),
    ],
)
def test_bad_incidents_are_errors(overrides, expected):
    rep = check(incidents(**overrides))
    assert not rep.ok
    assert any(expected in e for e in rep.errors), rep.errors


def test_missing_column_is_error():
    rep = check(incidents().drop(columns="priority"))
    assert not rep.ok and "missing columns" in rep.errors[0]


def test_soft_problems_are_warnings():
    df = incidents(category=["posting_error", "brand_new", "job_failure"],
                   resolution_notes=["x", None, "y"])  # fmt: skip
    rep = check(df)
    assert rep.ok
    assert any("category" in w for w in rep.warnings)
    assert any("resolution_notes" in w for w in rep.warnings)
    with pytest.raises(ValueError):
        check(incidents(module=["FI", "XX", "BASIS"])).raise_if_errors()


def signals():
    dates = pd.to_datetime(["2025-01-01", "2025-01-02"])
    return pd.DataFrame({
        "date": dates.repeat(3), "module": MODULES * 2,
        "job_failures": [0, 1, 4, 2, 0, 3], "idoc_errors": [1, 0, 0, 2, 1, 0],
        "change_events": [0, 0, 1, 0, 1, 0],
    })  # fmt: skip


def test_daily_signals():
    assert validate_daily_signals(signals(), MODULES, now=NOW).ok
    neg = signals()
    neg.loc[0, "job_failures"] = -1
    assert not validate_daily_signals(neg, MODULES, now=NOW).ok
    dup = pd.concat([signals(), signals().iloc[:1]])
    assert not validate_daily_signals(dup, MODULES, now=NOW).ok
    gap = signals().iloc[:-1]
    rep = validate_daily_signals(gap, MODULES, now=NOW)
    assert rep.ok and any("incomplete" in w for w in rep.warnings)


def flags(cal, day):
    row = cal.set_index("date").loc[pd.Timestamp(day)]
    return bool(row["is_month_end_close"]), bool(row["is_quarter_end"])


def test_month_end_close_definition():
    cal = build_calendar("2025-01-01", "2025-04-30")
    # Jan 2025: last 3 working days Wed 29 - Fri 31; Feb: first 2 working days Mon 3, Tue 4
    for d in ["2025-01-29", "2025-01-30", "2025-01-31", "2025-02-03", "2025-02-04"]:
        assert flags(cal, d) == (True, False), d
    for d in ["2025-01-28", "2025-02-01", "2025-02-02", "2025-02-05"]:
        assert flags(cal, d)[0] is False, d
    # March close is a quarter-end close, including the first working days of April
    for d in ["2025-03-27", "2025-03-28", "2025-03-31", "2025-04-01", "2025-04-02"]:
        assert flags(cal, d) == (True, True), d
    assert flags(cal, "2025-04-03") == (False, False)
    assert validate_calendar(cal).ok


def test_calendar_correct_at_range_edges():
    # A range that starts and ends mid-month must give the same flags as a full one.
    full = build_calendar("2025-03-01", "2025-04-30").set_index("date")
    part = build_calendar("2025-03-30", "2025-04-03").set_index("date")
    pd.testing.assert_frame_equal(part, full.loc[part.index])


def test_calendar_weekend_and_gaps():
    cal = build_calendar("2025-01-01", "2025-01-31")
    assert cal["is_weekend"].sum() == 8  # Jan 2025 has 4 Saturdays and 4 Sundays
    assert not validate_calendar(cal.drop(index=10)).ok
