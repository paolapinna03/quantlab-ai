"""Test della Strategy Discovery.

Il test chiave è `test_selection_ignores_confirmation_period`: se cambio
i prezzi del periodo di conferma, le candidate scelte non devono cambiare.
Altrimenti la "conferma" sarebbe una finzione.
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.discovery import DiscoveryConfig, full_space, run_discovery, sample_space
from quantlab.discovery.space import ENTRY_GRID
from quantlab.experiments import ExperimentRegistry
from quantlab.strategies.rules import ENTRIES, describe, rule_swing
from quantlab.swing import SwingConfig, run_swing_backtest


def make_ohlc(seed=0, assets=("AAA", "BBB", "CCC"), start="2005-01-03", end="2022-12-30"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    out = {}
    for a in assets:
        c = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, len(idx)))
        o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.002, len(idx)))
        out[a] = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.006,
                               "low": np.minimum(o, c) * 0.994, "close": c}, index=idx)
    return out


# --- Spazio di ricerca ---------------------------------------------------------------

def test_space_is_valid_and_reproducible():
    space = full_space()
    assert len(space) > 1000
    for params, engine in space:
        exits = (params["exit_rule"] == "opposite" or params["stop_atr"] > 0
                 or params["target_atr"] > 0 or engine.get("max_holding_days"))
        assert exits                                     # ogni candidata può uscire
    assert sample_space(20, seed=1) == sample_space(20, seed=1)
    assert sample_space(20, seed=1) != sample_space(20, seed=2)
    assert len({describe(p, e) for p, e in sample_space(50, 3)}) == 50   # nessun duplicato


@pytest.mark.parametrize("entry", ENTRIES)
def test_every_rule_is_causal(entry):
    data = make_ohlc(seed=5, assets=("X",), start="2010-01-01", end="2014-12-31")
    lb = ENTRY_GRID[entry]["lookback"][0]
    th = ENTRY_GRID[entry]["threshold"][-1]
    params = dict(entry=entry, lookback=lb, threshold=th, trend_filter=200, stop_atr=2.0)
    cut = data["X"].index[800]
    altered = {"X": data["X"].copy()}
    altered["X"].loc[cut + pd.Timedelta(days=1):] *= 1.7
    a = rule_swing(data, **params)["X"].loc[:cut]
    b = rule_swing(altered, **params)["X"].loc[:cut]
    pd.testing.assert_frame_equal(a, b)
    res = run_swing_backtest(data, rule_swing(data, **params), SwingConfig())
    assert res.equity.notna().all()


def test_unknown_entry_rejected():
    with pytest.raises(ValueError, match="entry sconosciuto"):
        rule_swing(make_ohlc(assets=("X",)), entry="magia")


# --- Ricerca in due fasi --------------------------------------------------------------

def small_config(**kw):
    base = dict(name="test", universe=("AAA", "BBB", "CCC"), budget=12, top_k=3, seed=3,
                min_trades=5, cash="zero")   # i test offline non scaricano i tassi
    return DiscoveryConfig(**(base | kw))


def test_discovery_two_stages(tmp_path):
    reg = ExperimentRegistry(tmp_path / "exp.sqlite")
    cfg = small_config()
    res = run_discovery(cfg, reg, ohlc=make_ohlc())
    assert len(res.search) == 12 and res.search["selected"].sum() == len(res.confirmed) <= 3
    # La fase di ricerca non ha mai visto dati dopo il training
    for rec in reg.records(cfg.search_family):
        if rec.status == "completed":
            assert pd.Timestamp(rec.eval_end) <= pd.Timestamp(cfg.train_end)
            assert reg.returns(rec.id).index[-1] <= pd.Timestamp(cfg.train_end)
    # La conferma valuta solo dal suo inizio, e ogni verdetto è salvato
    verdicts = reg.validations(cfg.confirm_family)
    for rec in reg.records(cfg.confirm_family):
        assert pd.Timestamp(rec.eval_start) >= pd.Timestamp(cfg.confirm_start)
    assert len(verdicts) == len(res.confirmed)
    assert set(verdicts) == set(res.confirmed["run_id"])
    assert reg.count_trials(cfg.search_family) == 12
    assert reg.discoveries()[0]["config"]["tested"] == 12
    # Ogni candidata confermata è giudicata contando TUTTE le K confermate
    k = len(res.confirmed)
    assert all(r.details["dsr"].n_trials == k for r in res.reports)


def test_selection_ignores_confirmation_period(tmp_path):
    cfg = small_config()
    base = make_ohlc()
    shocked = {a: d.copy() for a, d in base.items()}
    for d in shocked.values():
        d.loc["2017-01-01":] *= np.linspace(0.5, 2.0, len(d.loc["2017-01-01":]))[:, None]
    r1 = run_discovery(cfg, ExperimentRegistry(tmp_path / "a.sqlite"), ohlc=base)
    r2 = run_discovery(cfg, ExperimentRegistry(tmp_path / "b.sqlite"), ohlc=shocked)
    assert list(r1.confirmed["candidate"]) == list(r2.confirmed["candidate"])
    pd.testing.assert_series_equal(r1.search["sharpe"], r2.search["sharpe"])
    # ...mentre i risultati della conferma cambiano davvero
    assert not np.allclose(r1.confirmed["confirm_sharpe"], r2.confirmed["confirm_sharpe"])


def test_config_guards():
    with pytest.raises(ValueError, match="dopo la fine del training"):
        small_config(train_end="2017-06-30")
    with pytest.raises(ValueError, match="holdout"):
        small_config(confirm_end="2023-06-30")
