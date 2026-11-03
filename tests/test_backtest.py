"""Test del motore di backtest e delle prime strategie.

Il test più importante è `test_no_lookahead_*`: se modifichiamo i prezzi
*dopo* una certa data, i risultati *fino a* quella data non devono cambiare.
Se cambiano, la strategia o il motore stanno "sbirciando nel futuro".
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import BacktestConfig, run_backtest
from quantlab.strategies import buy_and_hold, time_series_momentum
from quantlab.strategies.base import period_end_mask

NO_COST = BacktestConfig(cost_bps=0.0)


def prices_from_returns(returns: dict, start="2024-01-01") -> pd.DataFrame:
    n = len(next(iter(returns.values())))
    idx = pd.bdate_range(start, periods=n + 1)
    data = {
        k: 100 * np.cumprod(np.concatenate([[1.0], 1 + np.asarray(v, dtype=float)]))
        for k, v in returns.items()
    }
    return pd.DataFrame(data, index=idx)


def random_prices(n=800, assets=("A", "B"), seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = {a: rng.normal(0.0003, 0.01, n) for a in assets}
    return prices_from_returns(r, start="2020-01-01")


def targets_at(prices, date_weights: dict) -> pd.DataFrame:
    t = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for i, w in date_weights.items():
        t.iloc[i] = w
    return t


# --- Tempistica ------------------------------------------------------------

def test_execution_lag_one_day():
    # Rendimenti dell'asset: giorno1 +1%, giorno2 +2%, giorno3 +3%
    prices = prices_from_returns({"A": [0.01, 0.02, 0.03]})
    # Segnale "compra tutto" alla chiusura del giorno 0
    res = run_backtest(prices, targets_at(prices, {0: [1.0]}), NO_COST)
    # Eseguito alla chiusura del giorno 1 → guadagna solo da giorno 2 in poi
    assert res.returns.iloc[1] == pytest.approx(0.0)
    assert res.returns.iloc[2] == pytest.approx(0.02)
    assert res.returns.iloc[3] == pytest.approx(0.03)


def test_execution_lag_zero_is_one_day_earlier():
    prices = prices_from_returns({"A": [0.01, 0.02, 0.03]})
    res = run_backtest(
        prices, targets_at(prices, {0: [1.0]}), BacktestConfig(cost_bps=0, execution_lag=0)
    )
    assert res.returns.iloc[1] == pytest.approx(0.01)


# --- Contabilità ------------------------------------------------------------

def test_fully_invested_matches_asset_returns():
    prices = random_prices(assets=("A",))
    res = run_backtest(prices, targets_at(prices, {0: [1.0]}), NO_COST)
    asset_r = prices["A"].pct_change()
    pd.testing.assert_series_equal(
        res.returns.iloc[2:], asset_r.iloc[2:], check_names=False
    )


def test_equity_is_compounded_returns():
    prices = random_prices()
    res = run_backtest(prices, buy_and_hold(prices))
    expected = 100_000 * (1 + res.returns).cumprod()
    pd.testing.assert_series_equal(res.equity, expected, check_names=False)


def test_weights_drift_without_rebalancing():
    # A raddoppia, B resta piatto. 50/50 iniziale → A pesa 2/3 dopo.
    prices = prices_from_returns({"A": [0.0, 1.0], "B": [0.0, 0.0]})
    res = run_backtest(prices, targets_at(prices, {0: [0.5, 0.5]}), NO_COST)
    assert res.weights.iloc[-1]["A"] == pytest.approx(2 / 3)
    assert res.returns.iloc[-1] == pytest.approx(0.5)


def test_cash_earns_nothing():
    prices = prices_from_returns({"A": [0.1, 0.1, 0.1]})
    res = run_backtest(prices, targets_at(prices, {0: [0.5]}), NO_COST)
    assert res.returns.iloc[2] == pytest.approx(0.05)


# --- Costi ------------------------------------------------------------------

def test_entry_cost():
    prices = prices_from_returns({"A": [0.0, 0.0, 0.0]})
    res = run_backtest(prices, targets_at(prices, {0: [1.0]}), BacktestConfig(cost_bps=10))
    # Turnover 1.0 alla prima esecuzione × 10 bp = 0.1%
    assert res.turnover.iloc[1] == pytest.approx(1.0)
    assert res.returns.iloc[1] == pytest.approx(-0.001)
    assert res.costs.sum() == pytest.approx(0.001)


def test_round_trip_cost():
    prices = prices_from_returns({"A": [0.0] * 4})
    t = targets_at(prices, {0: [1.0], 2: [0.0]})
    res = run_backtest(prices, t, BacktestConfig(cost_bps=10))
    assert res.turnover.sum() == pytest.approx(2.0)
    assert res.equity.iloc[-1] == pytest.approx(100_000 * 0.999 * 0.999)


def test_gross_and_net_differ_only_by_costs():
    prices = random_prices()
    res = run_backtest(prices, time_series_momentum(prices, lookback=60))
    no_cost = run_backtest(prices, time_series_momentum(prices, lookback=60), NO_COST)
    assert res.costs.sum() > 0
    pd.testing.assert_series_equal(res.gross_returns, no_cost.gross_returns)


# --- Controlli di sicurezza ---------------------------------------------------

def test_leverage_above_limit_is_rejected():
    prices = random_prices()
    with pytest.raises(ValueError, match="esposizione"):
        run_backtest(prices, targets_at(prices, {5: [1.0, 0.5]}))


def test_position_on_unlisted_asset_is_rejected():
    prices = random_prices()
    prices.iloc[:10, 1] = np.nan  # B non ancora quotato
    with pytest.raises(ValueError, match="senza prezzo"):
        run_backtest(prices, targets_at(prices, {2: [0.5, 0.5]}))


def test_unknown_asset_is_rejected():
    prices = random_prices()
    bad = pd.DataFrame({"ZZZ": 1.0}, index=prices.index[:1])
    with pytest.raises(ValueError, match="senza prezzi"):
        run_backtest(prices, bad)


# --- Nessun look-ahead -------------------------------------------------------

@pytest.mark.parametrize("lag", [0, 1, 2])
def test_no_lookahead_momentum(lag):
    prices = random_prices(n=900)
    cut = 600
    altered = prices.copy()
    altered.iloc[cut + 1 :] *= np.linspace(0.5, 3.0, len(prices) - cut - 1)[:, None]

    cfg = BacktestConfig(execution_lag=lag)
    a = run_backtest(prices, time_series_momentum(prices, lookback=120), cfg)
    b = run_backtest(altered, time_series_momentum(altered, lookback=120), cfg)
    # Fino a `cut` compreso i risultati devono essere identici
    pd.testing.assert_series_equal(a.returns.iloc[: cut + 1], b.returns.iloc[: cut + 1])
    # ...e dopo devono cambiare (altrimenti il test non verifica nulla)
    assert not a.returns.iloc[cut + 1 :].equals(b.returns.iloc[cut + 1 :])


# --- Strategie ----------------------------------------------------------------

def test_period_end_mask_marks_last_trading_day_of_month():
    idx = pd.bdate_range("2024-01-25", "2024-02-06")
    mask = period_end_mask(idx, "M")
    assert list(idx[mask]) == [pd.Timestamp("2024-01-31")]  # l'ultima data non conta


def test_momentum_long_in_uptrend_flat_in_downtrend():
    up = prices_from_returns({"UP": [0.001] * 300, "DOWN": [-0.001] * 300})
    w = time_series_momentum(up, lookback=100).dropna(how="all")
    assert (w["UP"] == 0.5).all()
    assert (w["DOWN"] == 0.0).all()


def test_momentum_long_short():
    up = prices_from_returns({"UP": [0.001] * 300, "DOWN": [-0.001] * 300})
    w = time_series_momentum(up, lookback=100, long_only=False).dropna(how="all")
    assert (w["DOWN"] == -0.5).all()


def test_momentum_rebalances_only_at_month_end():
    prices = random_prices(n=500)
    w = time_series_momentum(prices, lookback=60).dropna(how="all")
    assert (w.index.to_period("M").value_counts() == 1).all()


def test_buy_and_hold_waits_for_all_assets():
    prices = random_prices()
    prices.iloc[:30, 1] = np.nan
    w = buy_and_hold(prices).dropna(how="all")
    assert len(w) == 1
    assert w.index[0] == prices.index[30]
