"""Real-data adapter: maps a ServiceNow-style export to the schema, or skips cleanly."""

import pandas as pd
import pytest

from src.data.load_real import load_real, load_real_signals
from src.data.schema import INCIDENT_COLUMNS
from src.utils import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def export(**overrides):
    df = pd.DataFrame({
        "number": ["0001234", "0001235", "0001236", "0001237"],
        "opened_at": ["2025-03-03 09:15:00", "2025-03-03 11:00:00", "2025-03-04 08:30:00",
                      "2025-03-04 10:00:00"],
        "assignment_group": ["SAP FI Support", "SAP Basis", "Network Team", "SAP SD Support"],
        "priority": ["1 - Critical", "3 - Moderate", "4 - Low", "2 - High"],
        "subcategory": ["posting_error", "job_failure", "performance", "output_print"],
        "short_description": ["Period closed in FB50", "Job cancelled", "VPN slow",
                              "Invoice not printed"],
        "close_notes": ["Opened period", "Restarted job", "Rebooted router", None],
        "resolved_at": ["2025-03-03 10:00:00", None, "2025-03-05 09:00:00",
                        "2025-03-04 15:00:00"],
        "u_extra_field": ["a", "b", "c", "d"],
    })  # fmt: skip
    for k, v in overrides.items():
        df[k] = v
    return df


def test_missing_export_is_skipped(cfg, tmp_path, capsys):
    assert load_real(cfg, tmp_path / "nope.csv") is None
    assert load_real_signals(cfg, tmp_path / "nope.csv") is None
    assert "skipping" in capsys.readouterr().out


def test_export_is_mapped_to_schema(cfg, tmp_path):
    path = tmp_path / "incidents.csv"
    export().to_csv(path, index=False)
    df = load_real(cfg, path)
    assert list(df.columns) == list(INCIDENT_COLUMNS)
    assert len(df) == 3  # "Network Team" ticket is not an SAP module -> dropped
    assert df["module"].tolist() == ["FI", "BASIS", "SD"]
    assert df["priority"].tolist() == ["P1", "P3", "P2"]
    assert df["incident_id"].iloc[0] == "0001234"  # leading zeros kept
    assert pd.api.types.is_datetime64_any_dtype(df["created_at"])
    assert df["resolved_at"].isna().sum() == 1


def test_bad_dates_fail_validation(cfg, tmp_path):
    path = tmp_path / "incidents.csv"
    export(opened_at=["not a date", "2025-03-03 11:00:00", "2025-03-04", "2099-01-01"]).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="created_at"):
        load_real(cfg, path)


def test_export_without_required_column(cfg, tmp_path):
    path = tmp_path / "incidents.csv"
    export().drop(columns="priority").to_csv(path, index=False)
    with pytest.raises(ValueError, match="priority"):
        load_real(cfg, path)


def test_real_signals(cfg, tmp_path):
    modules = cfg["data"]["modules"]
    sig = pd.DataFrame({
        "date": ["2025-03-03"] * len(modules), "module": modules,
        "job_failures": 1, "idoc_errors": 0, "change_events": 0,
    })  # fmt: skip
    path = tmp_path / "signals.csv"
    sig.to_csv(path, index=False)
    out = load_real_signals(cfg, path)
    assert len(out) == len(modules)
