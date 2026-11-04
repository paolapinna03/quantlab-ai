"""Test dei metodi di allocazione: casi con soluzione nota a mano."""

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import BacktestConfig, run_backtest
from quantlab.portfolio import (
    METHODS,
    hrp,
    inverse_vol,
    ledoit_wolf,
    mean_variance,
    min_variance,
    risk_contributions,
    risk_parity,
)
from quantlab.strategies.allocation import risk_allocation

# Due asset non correlati: volatilità 10% e 20%
DIAG = np.diag([0.10**2, 0.20**2])


def correlated_cov(seed=0, n=5):
    """Covarianza valida (definita positiva): 2 fattori comuni + rischio specifico."""
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(n, 2))
    c = b @ b.T + np.diag(rng.uniform(0.5, 1.5, n))
    sd = np.sqrt(np.diag(c))
    corr = c / np.outer(sd, sd)
    vols = rng.uniform(0.05, 0.3, n)
    return corr * np.outer(vols, vols)


def test_inverse_vol_two_assets():
    # 1/0.10 : 1/0.20 = 2 : 1
    assert inverse_vol(DIAG) == pytest.approx([2 / 3, 1 / 3])


def test_min_variance_uncorrelated_is_inverse_variance():
    # Senza correlazione: w ∝ 1/σ² → 100 : 25 = 4 : 1
    assert min_variance(DIAG) == pytest.approx([0.8, 0.2], abs=1e-6)


def test_min_variance_respects_cap():
    w = min_variance(DIAG, max_weight=0.6)
    assert w == pytest.approx([0.6, 0.4], abs=1e-6)
    w5 = min_variance(correlated_cov(), max_weight=0.3)
    assert w5.max() <= 0.3 + 1e-6 and w5.sum() == pytest.approx(1)


def test_risk_parity_equalizes_risk_contributions():
    cov = correlated_cov(seed=3, n=6)
    w = risk_parity(cov)
    assert risk_contributions(w, cov) == pytest.approx(np.full(6, 1 / 6), abs=1e-5)
    # Senza correlazione coincide con inverse volatility
    assert risk_parity(DIAG) == pytest.approx(inverse_vol(DIAG), abs=1e-6)


def test_hrp_basic_properties():
    cov = correlated_cov(seed=4, n=7)
    w = hrp(cov)
    assert w.sum() == pytest.approx(1) and (w >= 0).all()
    assert hrp(DIAG) == pytest.approx([0.8, 0.2])   # due asset: inverso della varianza


def test_mean_variance_prefers_higher_return_and_respects_cap():
    w = mean_variance(np.array([0.10, 0.02]), DIAG, max_weight=0.7)
    assert w[0] == pytest.approx(0.7, abs=1e-5)


def test_ledoit_wolf_is_valid_and_shrinks_noisy_estimates():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.01, (40, 20))            # pochi dati, molti asset
    cov, delta = ledoit_wolf(x)
    assert 0 <= delta <= 1 and delta > 0.3
    assert np.linalg.eigvalsh(cov).min() > 0       # matrice definita positiva
    # Molti dati e volatilità diverse (lontane dal bersaglio): poco shrinkage
    many = rng.normal(0, [0.005, 0.01, 0.03], (5000, 3))
    assert ledoit_wolf(many)[1] < 0.05


# --- Strategia ---------------------------------------------------------------------

def make_prices(n=900, seed=7):
    rng = np.random.default_rng(seed)
    vols = np.array([0.006, 0.012, 0.02, 0.009])
    r = rng.normal(0.0003, vols, (n, 4))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0),
                        index=pd.bdate_range("2015-01-01", periods=n), columns=list("ABCD"))


@pytest.mark.parametrize("method", METHODS)
def test_allocation_rows_are_valid(method):
    w = risk_allocation(make_prices(), method=method, lookback=126).dropna(how="all")
    assert len(w) > 20
    assert np.allclose(w.sum(axis=1), 1) and (w >= 0).all().all()
    assert (w.index.to_period("M").value_counts() == 1).all()     # una riga al mese


@pytest.mark.parametrize("method", METHODS)
def test_allocation_has_no_lookahead(method):
    prices = make_prices()
    cut = 600
    altered = prices.copy()
    altered.iloc[cut + 1:] *= np.linspace(0.5, 2.0, len(prices) - cut - 1)[:, None]
    a = risk_allocation(prices, method=method, lookback=126).iloc[: cut + 1]
    b = risk_allocation(altered, method=method, lookback=126).iloc[: cut + 1]
    pd.testing.assert_frame_equal(a, b)


def test_momentum_filter_moves_losers_to_cash():
    prices = make_prices()
    prices["D"] = 100 * np.linspace(1, 0.4, len(prices))   # D scende sempre
    w = risk_allocation(prices, lookback=126, momentum_filter=252).dropna(how="all")
    assert (w["D"] == 0).all()
    assert (w.sum(axis=1) < 1).all()                       # la quota di D resta in liquidità


def test_unlisted_asset_is_excluded_then_enters():
    prices = make_prices()
    prices.iloc[:400, 3] = np.nan                          # D quotato dopo 400 giorni
    w = risk_allocation(prices, lookback=126).dropna(how="all")
    assert (w.loc[: prices.index[400 + 126 - 2], "D"] == 0).all()
    assert (w.iloc[-5:]["D"] > 0).all()
    run_backtest(prices, risk_allocation(prices, lookback=126), BacktestConfig())  # nessun errore
