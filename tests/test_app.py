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
    cfg = DiscoveryConfig(name="ui", universe=("SPY", "TLT"), budget=6, top_k=2, min_trades=3,
                          cash="zero")
    run_discovery(cfg, ExperimentRegistry(db), ohlc=ohlc)


def test_discovery_page_empty_state():
    at = AppTest.from_file(str(APP_DIR / "page_discovery.py"), default_timeout=60).run()
    assert not at.exception
    assert any(b.label == "Avvia la ricerca" for b in at.button)   # modulo aperto subito


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


def test_paper_page_empty_and_with_account(tmp_path, monkeypatch):
    from quantlab.paper import PaperStore, RiskLimits, create_account, run_account
    db = tmp_path / "paper.sqlite"
    monkeypatch.setenv("QUANTLAB_PAPER_DB", str(db))
    at = AppTest.from_file(str(APP_DIR / "page_paper.py"), default_timeout=60).run()
    assert not at.exception and any(b.label == "Crea il conto" for b in at.button)

    idx = pd.bdate_range("2019-01-01", periods=420)
    rng = np.random.default_rng(0)
    p = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, (420, 2)), axis=0),
                     index=idx, columns=["AAA", "BBB"])
    spec = {"strategy": "risk_allocation", "params": {"lookback": 126, "max_weight": 0.7},
            "universe": ["AAA", "BBB"], "cost_bps": 5.0, "cash": "zero"}
    store = PaperStore(db)
    create_account(store, "demo", spec, 10_000, RiskLimits(max_weight=0.7))
    for k in range(380, 420):
        run_account(store, "demo", prices=p.iloc[:k + 1], today=p.index[k])
    at = AppTest.from_file(str(APP_DIR / "page_paper.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.metric) == 5


# --- Azioni dall'interfaccia ---------------------------------------------------------

def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_data_page_renders_offline():
    at = AppTest.from_file(str(APP_DIR / "page_data.py"), default_timeout=60).run()
    assert not at.exception, at.exception


def test_discovery_form_rejects_bad_input():
    at = AppTest.from_file(str(APP_DIR / "page_discovery.py"), default_timeout=60).run()
    at.text_input[0].set_value("nome con spazi")
    _button(at, "Avvia la ricerca").click().run()          # casella di conferma non spuntata
    errors = " ".join(e.value for e in at.error)
    assert "conferma la casella" in errors and "solo lettere" in errors
    assert not ExperimentRegistry().discoveries()          # nessuna ricerca partita


@pytest.mark.network
def test_backtesting_page_all_strategies():
    at = AppTest.from_file(str(APP_DIR / "page_backtesting.py"), default_timeout=300).run()
    for key in ("paper", "momentum", "allocation", "rsi2", "donchian"):
        at.selectbox[0].set_value(key).run()
        assert not at.exception, (key, at.exception)
        assert len(at.metric) == 5


@pytest.mark.network
def test_discovery_form_runs_a_session():
    at = AppTest.from_file(str(APP_DIR / "page_discovery.py"), default_timeout=300).run()
    at.text_input[0].set_value("ui_test")
    at.number_input[0].set_value(10)
    at.number_input[1].set_value(2)
    at.multiselect[0].set_value(["SPY", "TLT", "GLD"])
    at.checkbox[0].check()
    _button(at, "Avvia la ricerca").click().run()
    assert not at.exception, at.exception
    assert ExperimentRegistry().discoveries()[0]["name"] == "ui_test"


@pytest.mark.network
def test_paper_create_and_run_from_ui(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTLAB_PAPER_DB", str(tmp_path / "paper.sqlite"))
    at = AppTest.from_file(str(APP_DIR / "page_paper.py"), default_timeout=300).run()
    at.text_input[0].set_value("ui_conto")
    _button(at, "Crea il conto").click().run()
    assert not at.exception, at.exception
    from quantlab.paper import PaperStore
    store = PaperStore(tmp_path / "paper.sqlite")
    assert store.last_day("ui_conto") is not None
    _button(at, "Elabora i giorni nuovi").click().run()   # nessun giorno nuovo: non duplica
    assert not at.exception and len(store.history("ui_conto")) == 1


def test_paper_page_compares_accounts(tmp_path, monkeypatch):
    from quantlab.paper import PaperStore, RiskLimits, create_account, run_account
    db = tmp_path / "paper.sqlite"
    monkeypatch.setenv("QUANTLAB_PAPER_DB", str(db))
    idx = pd.bdate_range("2019-01-01", periods=420)
    rng = np.random.default_rng(1)
    p = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, (420, 2)), axis=0),
                     index=idx, columns=["AAA", "BBB"])
    store = PaperStore(db)
    for name, spec in [("rp", {"strategy": "risk_allocation", "params": {"lookback": 126, "max_weight": 0.7}}),
                       ("mom", {"strategy": "time_series_momentum", "params": {"lookback": 126}})]:
        create_account(store, name, spec | {"universe": ["AAA", "BBB"], "cost_bps": 5.0, "cash": "zero"},
                       10_000, RiskLimits(max_weight=0.7))
        for k in range(380, 420):
            run_account(store, name, prices=p.iloc[:k + 1], today=p.index[k])
    at = AppTest.from_file(str(APP_DIR / "page_paper.py"), default_timeout=60).run()
    assert not at.exception, at.exception
