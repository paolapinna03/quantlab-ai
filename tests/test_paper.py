"""Test del paper trading.

Il test chiave è `test_paper_matches_backtest`: facendo girare il bot un giorno
alla volta, come dal vivo, i rendimenti devono coincidere con quelli del
motore di backtest. Così il track record del paper trading è confrontabile
con i risultati di ricerca.
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import BacktestConfig, run_backtest
from quantlab.data.rates import daily_cash_returns
from quantlab.paper import PaperStore, RiskLimits, create_account, halt, live_targets, resume, run_account
from quantlab.strategies import time_series_momentum
from quantlab.strategies.allocation import risk_allocation

IDX = pd.bdate_range("2019-01-01", periods=700)
YIELD = pd.Series(3.0, index=pd.bdate_range("2018-01-01", "2022-12-31"))


def prices(seed=0, crash=False):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0004, [0.006, 0.012, 0.009], (len(IDX), 3))
    if crash:
        r[450:470] = -0.03
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=IDX, columns=["AAA", "BBB", "CCC"])


SPEC = {"strategy": "risk_allocation",
        "params": {"method": "risk_parity", "lookback": 126, "max_weight": 0.6, "momentum_filter": 126},
        "universe": ["AAA", "BBB", "CCC"], "cost_bps": 5.0, "cash": "tbill"}


@pytest.fixture
def store(tmp_path):
    return PaperStore(tmp_path / "paper.sqlite")


def run_daily(store, name, p, start, end):
    for k in range(start, end + 1):
        run_account(store, name, prices=p.iloc[:k + 1], tbill_yield=YIELD, today=p.index[k])


# --- Calendario dal vivo ------------------------------------------------------------

def test_live_month_end_detection_matches_backtest():
    p = prices()
    full = risk_allocation(p, **SPEC["params"])
    for k in (300, 400, 500):
        live = live_targets(SPEC, p.iloc[:k + 1])
        pd.testing.assert_frame_equal(live, full.iloc[:k + 1], check_freq=False)  # anche l'ultimo giorno


# --- Coerenza con il backtest ---------------------------------------------------------

def test_paper_matches_backtest(store):
    p = prices()
    create_account(store, "x", SPEC, 10_000, RiskLimits(max_weight=0.6))
    start = 100                         # prima del warm-up: nessuna allocazione iniziale
    run_daily(store, "x", p, start, len(p) - 1)
    paper = store.history("x")["equity"].pct_change().dropna()

    rc = daily_cash_returns(YIELD, p.index)
    bt = run_backtest(p, risk_allocation(p, **SPEC["params"]), BacktestConfig(cost_bps=5.0), rc)
    expected = bt.returns.loc[paper.index]
    assert np.allclose(paper.to_numpy(), expected.to_numpy(), atol=1e-12)
    assert len(store.orders("x")) > 0


def test_catch_up_equals_daily_runs(tmp_path):
    p = prices(seed=3)
    a, b = PaperStore(tmp_path / "a.sqlite"), PaperStore(tmp_path / "b.sqlite")
    for s in (a, b):
        create_account(s, "x", SPEC, 10_000, RiskLimits(max_weight=0.6))
        run_account(s, "x", prices=p.iloc[:301], tbill_yield=YIELD, today=p.index[300])
    run_daily(a, "x", p, 301, 450)                                        # tutti i giorni
    run_account(b, "x", prices=p.iloc[:451], tbill_yield=YIELD, today=p.index[450])  # recupero in un colpo
    pd.testing.assert_frame_equal(a.history("x"), b.history("x"))


def test_running_twice_does_not_duplicate(store):
    p = prices()
    create_account(store, "x", SPEC, 10_000, RiskLimits(max_weight=0.6))
    run_daily(store, "x", p, 300, 320)
    n_days, n_orders = len(store.history("x")), len(store.orders("x"))
    s = run_account(store, "x", prices=p.iloc[:321], tbill_yield=YIELD, today=p.index[320])
    assert s.processed == [] and len(store.history("x")) == n_days
    assert len(store.orders("x")) == n_orders


def test_initial_allocation_uses_last_rebalance(store):
    p = prices()
    create_account(store, "x", SPEC, 10_000, RiskLimits(max_weight=0.6))
    run_account(store, "x", prices=p.iloc[:401], tbill_yield=YIELD, today=p.index[400])
    assert store.history("x")["cash"].iloc[0] == 10_000                 # primo giorno: tutto liquidità
    assert "allocazione_iniziale" in set(store.events("x")["kind"])
    run_account(store, "x", prices=p.iloc[:402], tbill_yield=YIELD, today=p.index[401])
    assert store.history("x")["cash"].iloc[-1] < 10_000                 # investito alla chiusura dopo


# --- Risk manager -------------------------------------------------------------------

def test_weight_above_limit_is_blocked(store):
    spec = SPEC | {"strategy": "time_series_momentum", "params": {"lookback": 60},
                   "universe": ["AAA"]}
    p = prices()[["AAA"]]
    create_account(store, "x", spec, 10_000, RiskLimits(max_weight=0.5))   # TSMOM su 1 asset vuole il 100%
    run_daily(store, "x", p, 200, 300)
    o = store.orders("x")
    assert len(o) > 0 and (o["status"] == "bloccato").all()
    assert (store.history("x")["cash"] > 0.99 * store.history("x")["equity"]).all()


def test_stale_data_blocks_trading(store):
    p = prices()
    create_account(store, "x", SPEC, 10_000, RiskLimits(max_weight=0.6))
    s = run_account(store, "x", prices=p.iloc[:301], tbill_yield=YIELD,
                    today=p.index[300] + pd.Timedelta(days=10))
    assert s.processed == [] and "dati_bloccati" in set(store.events("x")["kind"])


def test_drawdown_triggers_liquidation_and_halt(store):
    p = prices(crash=True)
    create_account(store, "x", SPEC, 10_000, RiskLimits(max_weight=0.6, halt_drawdown=-0.05))
    run_daily(store, "x", p, 300, 520)
    assert store.account("x")["status"] == "fermato"
    assert "arresto_automatico" in set(store.events("x")["kind"])
    h = store.history("x")
    assert h["cash"].iloc[-1] == pytest.approx(h["equity"].iloc[-1])    # tutto venduto


def test_manual_halt_and_resume(store):
    p = prices()
    create_account(store, "x", SPEC, 10_000, RiskLimits(max_weight=0.6))
    run_daily(store, "x", p, 300, 330)
    halt(store, "x", liquidate=True)
    run_daily(store, "x", p, 331, 400)
    h = store.history("x")
    assert h["cash"].iloc[-1] == pytest.approx(h["equity"].iloc[-1])
    n = len(store.orders("x"))
    run_daily(store, "x", p, 401, 460)                                  # fermato: nessun nuovo ordine
    assert len(store.orders("x")) == n
    resume(store, "x")
    run_daily(store, "x", p, 461, 560)
    assert len(store.orders("x")) > n


def test_swing_strategies_not_supported_yet(store):
    with pytest.raises(ValueError, match="swing"):
        create_account(store, "s", {"strategy": "donchian_breakout", "universe": ["AAA"]})
