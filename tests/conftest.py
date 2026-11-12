"""Shared test setup.

The tests check the main configuration (configs/config.yaml) and some use its dates, so a
SAP_CONFIG left set in the shell (e.g. after a quick run) must not leak into them. This runs
before any test module calls load_config() at import time.
"""

import os

os.environ.pop("SAP_CONFIG", None)

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def data_ready():
    """Skip (with the reason) tests that need the generated dataset when it is not there yet,
    e.g. in a fresh clone before `python run.py data`."""
    from src.data.build_dataset import load_daily, load_source
    from src.utils import load_config

    cfg = load_config()
    try:
        load_source(cfg)
        load_daily(cfg)
    except FileNotFoundError:
        pytest.skip("run `python run.py data` first")
