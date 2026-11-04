"""Test del rendimento della liquidità (T-bill) e dello Sharpe in eccesso."""

import math

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import BacktestConfig, evaluate, run_backtest
from quantlab.data.rates import daily_cash_returns, load_tbill_yield
from quantlab.experiments import ExperimentRegistry, ExperimentSpec, run_experiment
from quantlab.metrics import summarize
from quantlab.swing import SwingConfig, run_swing_backtest

IDX = pd.bdate_range("2020-01-01", periods=300)
FLAT_YIELD = pd.Series(5.0, index=pd.bdate_range("2019-01-01", "2021-12-31"))   # 5% annuo
DAILY_5 = (1.05) ** (1 / 252) - 1


def flat_prices(n=300) -> pd.DataFrame:
    return pd.DataFrame({"A": np.full(n, 100.0)}, index=IDX[:n])


# --- Tassi ---------------------------------------------------------------------

def test_daily_cash_return_from_annual_yield():
    rc = daily_cash_returns(FLAT_YIELD, IDX)
    assert rc.iloc[10] == pytest.approx(DAILY_5)
    assert (1 + rc.iloc[1:253]).prod() == pytest.approx(1.05, rel=1e-9)   # 252 giorni ≈ 1 anno


def test_cash_rate_uses_previous_day_and_floors_negative():
    y = pd.Series([2.0, -0.1, 3.0], index=pd.bdate_range("2020-03-02", periods=3))
    rc = daily_cash_returns(y, y.index)
    assert rc.iloc[0] == 0.0                                      # nessun dato precedente
    assert rc.iloc[1] == pytest.approx(1.02 ** (1 / 252) - 1)     # tasso di ieri, non di oggi
    assert rc.iloc[2] == 0.0                                      # -0,1% portato a zero


def test_rate_loader_rejects_wrong_scale(tmp_path):
    bad = pd.Series([450.0], index=pd.bdate_range("2020-01-01", periods=1))
    with pytest.raises(ValueError, match="fuori scala"):
        load_tbill_yield(cache_dir=tmp_path, fetcher=lambda: bad)


def test_rate_loader_caches(tmp_path):
    calls = []
    fetch = lambda: calls.append(1) or FLAT_YIELD
    load_tbill_yield(cache_dir=tmp_path, fetcher=fetch)
    s = load_tbill_yield(end="2020-06-30", cache_dir=tmp_path, fetcher=fetch)
    assert len(calls) == 1 and s.index[-1] <= pd.Timestamp("2020-06-30")


# --- Motori ---------------------------------------------------------------------

def test_portfolio_engine_cash_earns_rate():
    prices = flat_prices()
    targets = pd.DataFrame({"A": [0.5] + [np.nan] * 299}, index=IDX)
    rc = daily_cash_returns(FLAT_YIELD, IDX)
    res = run_backtest(prices, targets, BacktestConfig(cost_bps=0), rc)
    # Prima dell'ingresso tutto è in liquidità; il giorno dopo l'ingresso (peso 50%)
    # rende metà del tasso. Poi la quota di liquidità cresce, perché l'asset è piatto.
    assert res.returns.iloc[1] == pytest.approx(DAILY_5)
    assert res.returns.iloc[2] == pytest.approx(0.5 * DAILY_5)
    assert res.returns.iloc[50] > res.returns.iloc[2]
    no_cash = run_backtest(prices, targets, BacktestConfig(cost_bps=0))
    assert no_cash.returns.abs().max() == 0


def test_swing_engine_cash_earns_rate():
    df = pd.DataFrame({c: 100.0 for c in ("open", "high", "low", "close")}, index=IDX)
    sig = pd.DataFrame({"entry": False, "exit": False}, index=IDX)
    rc = daily_cash_returns(FLAT_YIELD, IDX)
    res = run_swing_backtest({"A": df}, {"A": sig}, SwingConfig(), rc)
    assert res.equity.iloc[-1] == pytest.approx(100_000 * (1 + rc).prod())


# --- Sharpe in eccesso ------------------------------------------------------------

def test_all_cash_strategy_has_no_sharpe():
    """Restare in liquidità non è un vantaggio: lo Sharpe in eccesso non è definito (zero rischio)."""
    rc = daily_cash_returns(FLAT_YIELD, IDX)
    s = summarize(rc, rf=rc)
    assert math.isnan(s["sharpe"])
    assert s["cagr"] == pytest.approx(0.05, rel=1e-2)            # ma il rendimento c'è
    assert s["cash_rate"] == pytest.approx(0.05, rel=1e-2)


def test_excess_sharpe_is_lower_than_raw_sharpe():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 300), index=IDX)
    rc = daily_cash_returns(FLAT_YIELD, IDX)
    assert summarize(r, rf=rc)["sharpe"] < summarize(r)["sharpe"]


def test_evaluate_uses_cash_rate_of_the_backtest():
    rng = np.random.default_rng(1)
    prices = pd.DataFrame({"A": 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, 300))}, index=IDX)
    targets = pd.DataFrame({"A": [1.0] + [np.nan] * 299}, index=IDX)
    rc = daily_cash_returns(FLAT_YIELD, IDX)
    with_cash = evaluate(run_backtest(prices, targets, BacktestConfig(cost_bps=0), rc))
    without = evaluate(run_backtest(prices, targets, BacktestConfig(cost_bps=0)))
    assert with_cash["sharpe"] < without["sharpe"]                # stesso portafoglio, asticella più alta


# --- Esperimenti ------------------------------------------------------------------

def test_cash_option_in_spec(tmp_path):
    prices = pd.DataFrame({"SPY": 100 * np.cumprod(1 + np.random.default_rng(2).normal(
        0.0003, 0.01, 1500))}, index=pd.bdate_range("2014-01-01", periods=1500))
    reg = ExperimentRegistry(tmp_path / "r.sqlite")
    base = ExperimentSpec(family="c", strategy="time_series_momentum", params={"lookback": 126})
    zero = run_experiment(base, reg, prices=prices)
    tbill = run_experiment(ExperimentSpec(**{**base.to_dict(), "universe": ("SPY",), "cash": "tbill"}),
                           reg, prices=prices, tbill_yield=FLAT_YIELD.reindex(
                               pd.bdate_range("2013-01-01", "2020-12-31")).ffill().bfill())
    assert tbill.metrics["cagr"] > zero.metrics["cagr"]          # la liquidità ora rende
    assert "^IRX" in reg.get(tbill.run_id).data                  # tasso nell'impronta dei dati
    assert reg.count_trials("c") == 2                            # ipotesi diversa = tentativo diverso
    with pytest.raises(ValueError, match="cash deve essere"):
        run_experiment(ExperimentSpec(family="c", strategy="buy_and_hold", cash="bond"), prices=prices)


def test_zero_cash_keeps_old_hashes():
    spec = ExperimentSpec(family="f", strategy="buy_and_hold")
    assert "cash" not in spec.to_dict() or spec.cash == "zero"
    assert spec.config_hash() == ExperimentSpec(family="f", strategy="buy_and_hold", cash="zero").config_hash()
    assert spec.config_hash() != ExperimentSpec(family="f", strategy="buy_and_hold", cash="tbill").config_hash()
