"""Test del motore swing, degli indicatori e delle strategie swing.

I casi del motore usano barre costruite a mano, così ogni prezzo di
esecuzione si può verificare con una calcolatrice.
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.experiments import ExperimentRegistry, ExperimentSpec, run_experiment
from quantlab.indicators import atr, donchian_high_prior, rsi, sma
from quantlab.strategies.swing import donchian_breakout, rsi2_mean_reversion
from quantlab.swing import SwingConfig, run_swing_backtest, trade_stats

NO_FRICTION = dict(cost_bps=0.0, slippage_bps=0.0)


def bars(rows, start="2024-01-01") -> pd.DataFrame:
    """rows = [(open, high, low, close), ...]"""
    idx = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"], dtype=float)


def signals(df, entry=(), exit=(), stop=np.nan, target=np.nan, score=np.nan) -> pd.DataFrame:
    s = pd.DataFrame(index=df.index)
    s["entry"] = [i in entry for i in range(len(df))]
    s["exit"] = [i in exit for i in range(len(df))]
    s["stop_distance"] = stop
    s["target_distance"] = target
    s["score"] = score
    return s


FLAT = [(100, 101, 99, 100)] * 2


# --- Indicatori ---------------------------------------------------------------

def test_sma_and_rsi_basics():
    s = pd.Series([1.0, 2, 3, 4, 5, 6])
    assert sma(s, 3).iloc[-1] == pytest.approx(5.0)
    assert sma(s, 3).iloc[:2].isna().all()
    assert rsi(s, 2).iloc[-1] == pytest.approx(100.0)      # solo rialzi
    down = pd.Series([6.0, 5, 4, 3, 2, 1])
    assert rsi(down, 2).iloc[-1] == pytest.approx(0.0)    # solo ribassi


def test_atr_of_constant_range():
    n = 30
    df = bars([(100, 102, 98, 100)] * n)
    assert atr(df["high"], df["low"], df["close"], 14).iloc[-1] == pytest.approx(4.0)


def test_donchian_excludes_today():
    high = pd.Series([1.0, 2, 3, 10])
    assert donchian_high_prior(high, 3).iloc[-1] == 3.0   # il 10 di oggi non conta


def test_indicators_are_causal():
    rng = np.random.default_rng(0)
    c = pd.Series(100 + rng.normal(0, 1, 300).cumsum())
    changed = c.copy()
    changed.iloc[200:] += 50
    for f in (lambda x: sma(x, 20), lambda x: rsi(x, 2), lambda x: donchian_high_prior(x, 55)):
        pd.testing.assert_series_equal(f(c).iloc[:200], f(changed).iloc[:200])


# --- Esecuzione --------------------------------------------------------------------

def test_entry_and_exit_at_next_open():
    df = bars([(100, 101, 99, 100), (102, 104, 101, 103), (105, 106, 104, 105), (107, 108, 106, 107)])
    sig = signals(df, entry=[0], exit=[2])
    res = run_swing_backtest({"A": df}, {"A": sig}, SwingConfig(position_fraction=1.0, **NO_FRICTION))
    t = res.trades.iloc[0]
    assert t["entry_price"] == 102 and t["entry_date"] == df.index[1]   # apertura del giorno dopo
    assert t["exit_price"] == 107 and t["exit_date"] == df.index[3]
    assert t["return"] == pytest.approx(107 / 102 - 1)


def test_costs_and_slippage():
    df = bars([(100, 100, 100, 100)] * 4)
    sig = signals(df, entry=[0], exit=[1])
    cfg = SwingConfig(position_fraction=1.0, cost_bps=10, slippage_bps=20)
    t = run_swing_backtest({"A": df}, {"A": sig}, cfg).trades.iloc[0]
    assert t["entry_price"] == pytest.approx(100.2)     # paghi di più comprando
    assert t["exit_price"] == pytest.approx(99.8)       # incassi meno vendendo
    v = 100_000 / 1.001                                  # capitale investito dopo la commissione
    expected = (v / 100.2 * 99.8 * 0.999 - 100_000) / 100_000
    assert t["return"] == pytest.approx(expected)


def test_stop_inside_bar_fills_at_stop():
    df = bars([(100, 100, 100, 100), (100, 101, 99, 100), (99, 100, 94, 96)])
    sig = signals(df, entry=[0], stop=5.0)
    t = run_swing_backtest({"A": df}, {"A": sig}, SwingConfig(**NO_FRICTION)).trades.iloc[0]
    assert t["exit_reason"] == "stop" and t["exit_price"] == pytest.approx(95.0)


def test_gap_through_stop_fills_at_open():
    df = bars([(100, 100, 100, 100), (100, 101, 99, 100), (90, 91, 88, 89)])
    sig = signals(df, entry=[0], stop=5.0)
    t = run_swing_backtest({"A": df}, {"A": sig}, SwingConfig(**NO_FRICTION)).trades.iloc[0]
    assert t["exit_price"] == pytest.approx(90.0)        # peggio dello stop a 95


def test_target_and_ambiguous_bar():
    df = bars([(100, 100, 100, 100), (100, 101, 99, 100), (100, 111, 99, 108)])
    t = run_swing_backtest({"A": df}, {"A": signals(df, entry=[0], target=10.0)},
                           SwingConfig(**NO_FRICTION)).trades.iloc[0]
    assert t["exit_reason"] == "target" and t["exit_price"] == pytest.approx(110.0)
    both = bars([(100, 100, 100, 100), (100, 101, 99, 100), (100, 111, 94, 100)])
    t = run_swing_backtest({"A": both}, {"A": signals(both, entry=[0], stop=5.0, target=10.0)},
                           SwingConfig(**NO_FRICTION)).trades.iloc[0]
    assert t["exit_reason"] == "stop"                    # ipotesi prudente


def test_time_stop():
    df = bars([(100, 101, 99, 100)] * 8)
    res = run_swing_backtest({"A": df}, {"A": signals(df, entry=[0])},
                             SwingConfig(max_holding_days=3, **NO_FRICTION))
    t = res.trades.iloc[0]
    assert t["bars_held"] == 3 and t["exit_date"] == df.index[4]


def test_stale_exit_order_does_not_hit_new_position():
    # Giorno 2: lo stop chiude la posizione, ma c'era anche un segnale di uscita.
    # Il giorno 3 si rientra: la vecchia uscita non deve chiudere il nuovo trade.
    df = bars([(100, 100, 100, 100), (100, 101, 99, 100), (99, 100, 90, 95),
               (96, 97, 95, 96), (96, 97, 95, 96), (96, 97, 95, 96)])
    sig = signals(df, entry=[0, 2], exit=[1], stop=5.0)
    res = run_swing_backtest({"A": df}, {"A": sig}, SwingConfig(**NO_FRICTION))
    assert len(res.trades) == 1 and len(res.open_positions) == 1


# --- Portafoglio ------------------------------------------------------------------

def test_max_positions_and_priority():
    data = {a: bars(FLAT * 3) for a in ("A", "B", "C")}
    sig = {"A": signals(data["A"], entry=[0], score=1.0),
           "B": signals(data["B"], entry=[0], score=3.0),
           "C": signals(data["C"], entry=[0], score=2.0)}
    res = run_swing_backtest(data, sig, SwingConfig(max_positions=2, **NO_FRICTION))
    assert sorted(res.open_positions["asset"]) == ["B", "C"]   # i due punteggi più alti


def test_no_leverage_and_equity_reconciles():
    rng = np.random.default_rng(3)
    data, sig = {}, {}
    for a in ("A", "B", "C", "D"):
        c = 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, 600))
        o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.003, 600))
        df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01,
                           "low": np.minimum(o, c) * 0.99, "close": c},
                          index=pd.bdate_range("2015-01-01", periods=600))
        data[a] = df
        s = signals(df, entry=list(range(0, 600, 7)), exit=list(range(3, 600, 7)), stop=3.0)
        sig[a] = s
    cfg = SwingConfig(max_positions=4, position_fraction=0.5)
    res = run_swing_backtest(data, sig, cfg)
    assert (res.weights.sum(axis=1) <= 1 + 1e-9).all()
    unrealized = sum(p.shares * data[p.asset]["close"].iloc[-1] for p in res.open_positions.itertuples())
    cash_invested = sum(p.shares * p.entry_price * (1 + cfg.cost_bps / 1e4)
                        for p in res.open_positions.itertuples())
    expected = 100_000 + res.trades["pnl"].sum() - cash_invested + unrealized
    assert res.equity.iloc[-1] == pytest.approx(expected)


def test_risk_sizing():
    df = bars([(100, 100, 100, 100)] * 3)
    cfg = SwingConfig(sizing="risk", risk_per_trade=0.01, position_fraction=1.0, **NO_FRICTION)
    res = run_swing_backtest({"A": df}, {"A": signals(df, entry=[0], stop=4.0)}, cfg)
    p = res.open_positions.iloc[0]
    assert p["shares"] * 4.0 == pytest.approx(1_000)    # perdita allo stop = 1% di 100.000


def test_trade_stats():
    t = pd.DataFrame({"return": [0.10, -0.05, 0.02, -0.01], "pnl": [100, -50, 20, -10],
                      "bars_held": [3, 2, 5, 1], "exit_reason": ["signal", "stop", "target", "stop"]})
    s = trade_stats(t)
    assert s["win_rate"] == 0.5
    assert s["profit_factor"] == pytest.approx(120 / 60)
    assert s["payoff_ratio"] == pytest.approx(0.06 / 0.03)
    assert s["expectancy"] == pytest.approx(0.015)
    assert s["exits_stop"] == 2


# --- Nessun look-ahead ---------------------------------------------------------------

@pytest.mark.parametrize("strategy", [rsi2_mean_reversion, donchian_breakout])
def test_no_lookahead(strategy):
    rng = np.random.default_rng(9)
    n = 900
    c = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, n))
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.008,
                       "low": np.minimum(o, c) * 0.992, "close": c},
                      index=pd.bdate_range("2010-01-01", periods=n))
    cut = 600
    altered = df.copy()
    altered.iloc[cut + 1:] *= np.linspace(0.6, 1.8, n - cut - 1)[:, None]
    cfg = SwingConfig()
    a = run_swing_backtest({"X": df}, strategy({"X": df}), cfg)
    b = run_swing_backtest({"X": altered}, strategy({"X": altered}), cfg)
    pd.testing.assert_series_equal(a.equity.iloc[: cut + 1], b.equity.iloc[: cut + 1])
    done_a = a.trades[a.trades["exit_date"] <= df.index[cut]]
    done_b = b.trades[b.trades["exit_date"] <= df.index[cut]]
    pd.testing.assert_frame_equal(done_a.reset_index(drop=True), done_b.reset_index(drop=True))
    assert len(done_a) > 0


# --- Registro --------------------------------------------------------------------

def test_swing_runs_through_registry(tmp_path):
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2010-01-01", "2022-12-30")
    ohlc = {}
    for a in ("AAA", "BBB"):
        c = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, len(idx)))
        o = np.r_[c[0], c[:-1]]
        ohlc[a] = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01,
                                "low": np.minimum(o, c) * 0.99, "close": c}, index=idx)
    reg = ExperimentRegistry(tmp_path / "exp.sqlite")
    spec = ExperimentSpec(family="swing_test", strategy="donchian_breakout",
                          universe=("AAA", "BBB"), engine={"max_positions": 2})
    out = run_experiment(spec, reg, ohlc=ohlc)
    assert out.run_id and out.metrics["trade_trades"] > 0
    assert out.eval_start > idx[55]                       # dopo il warm-up
    with pytest.raises(ValueError, match="motore swing"):
        run_experiment(ExperimentSpec(family="swing_test", strategy="donchian_breakout",
                                      universe=("AAA",), engine={"leva": 3}), reg, ohlc=ohlc)
    with pytest.raises(ValueError, match="solo per le strategie swing"):
        run_experiment(ExperimentSpec(family="x", strategy="buy_and_hold", engine={"max_positions": 2}))
