"""Smoke test della dashboard: le pagine partono senza errori e mostrano numeri reali.

I test che scaricano dati veri girano solo con:
    pytest -m network
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from quantlab.discovery import DiscoveryConfig, run_discovery
from quantlab.discovery.report import load_session
from quantlab.experiments import ExperimentRegistry

APP_DIR = Path(__file__).resolve().parents[1] / "app"


@pytest.fixture(autouse=True)
def temp_registry(tmp_path, monkeypatch):
    """I test non devono scrivere nel registro reale degli esperimenti."""
    db = tmp_path / "exp.sqlite"
    monkeypatch.setenv("QUANTLAB_DB", str(db))
    return db


def synthetic_discovery(db) -> None:
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2005-01-03", "2022-12-30")
    ohlc = {}
    for t in ("SPY", "TLT"):
        c = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, len(idx)))
        o = np.r_[c[0], c[:-1]]
        ohlc[t] = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.006,
                                "low": np.minimum(o, c) * 0.994, "close": c}, index=idx)
    cfg = DiscoveryConfig(name="ui", universe=("SPY", "TLT"), budget=6, top_k=2, min_trades=3)
    run_discovery(cfg, ExperimentRegistry(db), ohlc=ohlc)


def test_discovery_page_empty_state():
    at = AppTest.from_file(str(APP_DIR / "page_discovery.py"), default_timeout=60).run()
    assert not at.exception
    assert any("python -m quantlab.discovery" in i.value for i in at.info)


def test_load_session_matches_discovery(temp_registry):
    synthetic_discovery(temp_registry)
    s = load_session(ExperimentRegistry(temp_registry), "ui")
    assert len(s["search"]) == 6
    assert s["search"]["selected"].sum() == len(s["confirmed"])
    assert set(s["confirmed"]["verdict"]) <= {"PROMOSSA", "RESPINTA"}


@pytest.mark.network
def test_discovery_page_with_results(temp_registry):
    synthetic_discovery(temp_registry)
    at = AppTest.from_file(str(APP_DIR / "page_discovery.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.metric) == 4


@pytest.mark.network
def test_dashboard_runs_without_errors():
    at = AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.metric) == 5
    assert at.metric[0].label == "CAGR"


@pytest.mark.network
def test_dashboard_warns_when_parameters_change():
    at = AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=120).run()
    assert not at.warning
    at.select_slider[0].set_value(126).run()
    assert not at.exception
    assert len(at.warning) == 1
