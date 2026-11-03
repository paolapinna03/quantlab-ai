"""Test delle metriche: ogni valore atteso è calcolato a mano nei commenti.

Se un giorno una di queste metriche cambia per errore, tutti i backtest e i
confronti tra strategie diventerebbero sbagliati senza che nessuno se ne
accorga. Questi test sono la nostra rete di sicurezza.
"""

import math

import numpy as np
import pandas as pd
import pytest

from quantlab.metrics import (
    annualized_volatility,
    cagr,
    calmar_ratio,
    drawdown_series,
    hit_rate,
    max_drawdown,
    max_drawdown_duration,
    profit_factor,
    sharpe_ratio,
    sharpe_standard_error,
    sortino_ratio,
    summarize,
    total_return,
)


def series(values) -> pd.Series:
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=idx, dtype="float64")


# Serie di riferimento usata in più test:
#   r = [+2%, 0%, +1%, -1%]
#   media = 0.005
#   scarti dalla media = [0.015, -0.005, 0.005, -0.015]
#   somma quadrati = 0.000225 + 0.000025 + 0.000025 + 0.000225 = 0.0005
#   varianza (ddof=1) = 0.0005 / 3
REF = series([0.02, 0.0, 0.01, -0.01])
REF_STD = math.sqrt(0.0005 / 3)


def test_total_return():
    # 1.02 * 1.00 * 1.01 * 0.99 - 1
    assert total_return(REF) == pytest.approx(1.02 * 1.01 * 0.99 - 1)


def test_cagr_doubling_over_two_years():
    # Raddoppio in 504 giorni (2 anni): (2)^(1/2) - 1 = 41.42% annuo
    daily = 2 ** (1 / 504) - 1
    r = series([daily] * 504)
    assert cagr(r) == pytest.approx(math.sqrt(2) - 1)


def test_cagr_total_loss():
    assert cagr(series([0.1, -1.0])) == -1.0


def test_volatility():
    assert annualized_volatility(REF) == pytest.approx(REF_STD * math.sqrt(252))


def test_sharpe_zero_risk_free():
    # 0.005 / 0.01291 * sqrt(252) ≈ 6.15
    assert sharpe_ratio(REF) == pytest.approx(0.005 / REF_STD * math.sqrt(252))


def test_sharpe_with_risk_free_reduces_ratio():
    assert sharpe_ratio(REF, risk_free=0.05) < sharpe_ratio(REF)


def test_sharpe_undefined_without_volatility():
    assert math.isnan(sharpe_ratio(series([0.01] * 10)))


def test_sortino():
    # Solo -1% è sotto il target 0: media dei quadrati = 0.0001 / 4 = 0.000025
    # deviazione al ribasso = 0.005
    # Sortino = 0.005 * 252 / (0.005 * sqrt(252)) = sqrt(252) ≈ 15.87
    assert sortino_ratio(REF) == pytest.approx(math.sqrt(252))


def test_sharpe_standard_error_shrinks_with_more_data():
    rng = np.random.default_rng(0)
    short = series(rng.normal(0.0004, 0.01, 252))
    long_ = series(rng.normal(0.0004, 0.01, 252 * 16))
    assert sharpe_standard_error(long_) < sharpe_standard_error(short)
    # Con 16x dati l'errore si riduce di circa sqrt(16) = 4 volte
    ratio = sharpe_standard_error(short) / sharpe_standard_error(long_)
    assert ratio == pytest.approx(4, rel=0.1)


def test_max_drawdown():
    # Capitale: 1 → 1.10 → 0.55 → 0.66
    # Massimo 1.10, minimo successivo 0.55: 0.55 / 1.10 - 1 = -50%
    r = series([0.10, -0.50, 0.20])
    assert max_drawdown(r) == pytest.approx(-0.50)


def test_drawdown_counts_loss_on_first_day():
    # Si parte da 1.0; perdere il 10% il primo giorno è già un drawdown
    assert max_drawdown(series([-0.10, 0.05])) == pytest.approx(-0.10)


def test_drawdown_zero_when_always_rising():
    assert max_drawdown(series([0.01, 0.02, 0.01])) == 0.0
    assert (drawdown_series(series([0.01, 0.02])) == 0).all()


def test_max_drawdown_duration():
    # Capitale: 1.1, 0.99, 1.04, 1.144 (nuovo massimo), 1.03
    # Sotto il massimo nei periodi 2 e 3 (2 periodi), poi di nuovo nel 5 (1)
    r = series([0.10, -0.10, 0.05, 0.10, -0.10])
    assert max_drawdown_duration(r) == 2


def test_calmar():
    r = series([0.10, -0.50, 0.20])
    assert calmar_ratio(r) == pytest.approx(cagr(r) / 0.50)


def test_hit_rate_ignores_flat_days():
    # 2 positivi, 1 negativo, 1 piatto (escluso) → 2/3
    assert hit_rate(REF) == pytest.approx(2 / 3)


def test_profit_factor():
    # guadagni 0.03, perdite 0.01 → 3
    assert profit_factor(REF) == pytest.approx(3.0)


def test_leading_nan_is_ignored():
    with_nan = pd.concat([series([np.nan]), REF])
    with_nan.index = pd.bdate_range("2023-12-29", periods=5)
    assert sharpe_ratio(with_nan) == pytest.approx(sharpe_ratio(REF))


def test_summarize_has_all_metrics():
    s = summarize(REF)
    for key in ["cagr", "sharpe", "sortino", "max_drawdown", "calmar", "sharpe_se"]:
        assert key in s.index
    assert s["periods"] == 4


# --- Controllo di realtà con dati veri --------------------------------------

@pytest.mark.network
def test_spy_global_financial_crisis_drawdown(tmp_path):
    """SPY ha perso circa il 55% (total return) tra ottobre 2007 e marzo 2009."""
    from quantlab.data import load_prices
    from quantlab.metrics import returns_from_prices

    prices = load_prices("SPY", "2007-01-01", "2009-12-31", cache_dir=tmp_path)
    mdd = max_drawdown(returns_from_prices(prices["close"]))
    assert -0.58 < mdd < -0.50
