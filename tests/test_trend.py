"""Test di trend su più orizzonti e volatility targeting."""

import numpy as np
import pandas as pd
import pytest

from quantlab.experiments import ExperimentSpec
from quantlab.strategies.catalog import warmup_days
from quantlab.strategies.momentum import time_series_momentum
from quantlab.strategies.trend import trend_signals

IDX = pd.bdate_range("2015-01-01", periods=800)


def prices(vols=(0.01, 0.01, 0.01), drifts=(0.001, -0.001, 0.0005), seed=1):
    rng = np.random.default_rng(seed)
    r = rng.normal(drifts, vols, (len(IDX), len(vols)))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=IDX, columns=list("ABC")[:len(vols)])


def test_single_horizon_matches_time_series_momentum():
    p = prices()
    a = trend_signals(p, horizons=(252,))
    b = time_series_momentum(p, lookback=252)
    rows = a.dropna(how="all").index
    pd.testing.assert_frame_equal(a.loc[rows], b.loc[rows].fillna(0.0))


def test_multi_horizon_gives_fractional_exposure():
    w = trend_signals(prices(), horizons=(21, 63, 126, 252)).dropna(how="all")
    allowed = {0.0, 0.25 / 3, 0.5 / 3, 0.75 / 3, 1 / 3}
    assert set(np.round(w.to_numpy().ravel(), 10)) <= {round(x, 10) for x in allowed}
    assert (w.sum(axis=1) <= 1 + 1e-12).all()


def test_vol_target_scales_down_never_up():
    p = prices(vols=(0.03, 0.03, 0.03), drifts=(0.001, 0.001, 0.001))
    plain = trend_signals(p).dropna(how="all")
    capped = trend_signals(p, vol_target=0.10).dropna(how="all")
    assert (capped.sum(axis=1) <= plain.sum(axis=1) + 1e-12).all()
    assert (capped.sum(axis=1) < plain.sum(axis=1)).any()
    calm = prices(vols=(0.001, 0.001, 0.001), drifts=(0.001, 0.001, 0.001))
    pd.testing.assert_frame_equal(trend_signals(calm, vol_target=0.10), trend_signals(calm))


def test_no_look_ahead():
    p = prices()
    full = trend_signals(p, horizons=(21, 252), vol_target=0.08)
    cut = trend_signals(p.iloc[:600], horizons=(21, 252), vol_target=0.08)
    common = cut.dropna(how="all").index
    pd.testing.assert_frame_equal(full.loc[common], cut.loc[common])


def test_warmup_and_hash():
    assert warmup_days("trend_signals", {"horizons": (21, 252)}) == 252
    assert warmup_days("trend_signals", {"horizons": (21,), "vol_target": 0.1, "vol_lookback": 63}) == 63
    a = ExperimentSpec(family="f", strategy="trend_signals", params={"horizons": (21, 252)})
    b = ExperimentSpec(family="f", strategy="trend_signals", params={"horizons": [21, 252]})
    assert a.config_hash() == b.config_hash()


@pytest.mark.parametrize("kw", [{"horizons": ()}, {"vol_target": -0.1}, {"vol_target": 0.1, "vol_lookback": 5}])
def test_invalid_params(kw):
    with pytest.raises(ValueError):
        trend_signals(prices(), **kw)
