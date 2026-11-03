"""Test della validazione statistica.

Il test più istruttivo è `test_selection_bias_is_caught`: 50 strategie
puramente casuali, scelgo la migliore. Il suo Sharpe sembra significativo
(PSR alto), ma il Deflated Sharpe, sapendo che ho fatto 50 tentativi, la boccia.
"""

import math

import numpy as np
import pandas as pd
import pytest

from quantlab.experiments import ExperimentRegistry, ExperimentSpec
from quantlab.metrics import sharpe_ratio
from quantlab.validation import (
    Criteria,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_difference_test,
    validate_candidate,
    walk_forward,
)


def noise(n=1260, mu=0.0, sigma=0.01, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n), index=pd.bdate_range("2010-01-01", periods=n))


# --- Probabilistic Sharpe Ratio ----------------------------------------------------

def test_psr_is_half_at_its_own_sharpe():
    r = noise(mu=0.0005)
    sr_annual = r.mean() / r.std() * math.sqrt(252)
    assert probabilistic_sharpe_ratio(r, sr_annual) == pytest.approx(0.5)


def test_psr_matches_formula():
    r = noise(mu=0.0004, seed=3)
    sr = r.mean() / r.std()
    g3, g4 = r.skew(), r.kurt() + 3
    z = sr * math.sqrt(len(r) - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    expected = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    assert probabilistic_sharpe_ratio(r) == pytest.approx(expected)


def test_psr_grows_with_sample_length():
    short = noise(n=252, mu=0.002, seed=1)
    assert short.mean() > 0  # con Sharpe stimato positivo, più dati = più fiducia
    long_ = pd.Series(np.tile(short.values, 8), index=pd.bdate_range("2000-01-01", periods=252 * 8))
    assert probabilistic_sharpe_ratio(long_) > probabilistic_sharpe_ratio(short)


def test_negative_skew_lowers_psr():
    rng = np.random.default_rng(5)
    base = rng.normal(0.0006, 0.008, 2000)
    crashes = np.where(rng.random(2000) < 0.01, -0.06, 0.0)   # rari crolli
    skewed = pd.Series(base + crashes + 0.0006, index=pd.bdate_range("2010-01-01", periods=2000))
    assert skewed.skew() < -1
    # Serie simmetrica con ESATTAMENTE la stessa media e deviazione standard
    sym = pd.Series(np.random.default_rng(1).normal(0, 1, 2000), index=skewed.index)
    sym = (sym - sym.mean()) / sym.std() * skewed.std() + skewed.mean()
    # Stesso Sharpe, ma code sinistre più pesanti → fiducia minore
    assert probabilistic_sharpe_ratio(skewed) < probabilistic_sharpe_ratio(sym)


# --- Deflated Sharpe Ratio ---------------------------------------------------------

def test_expected_max_sharpe_properties():
    assert expected_max_sharpe(1, 0.5) == 0.0
    assert expected_max_sharpe(10, 0.5) < expected_max_sharpe(100, 0.5)
    assert expected_max_sharpe(10, 1.0) == pytest.approx(2 * expected_max_sharpe(10, 0.5))


def test_dsr_with_one_trial_equals_psr():
    r = noise(mu=0.0004)
    ds = deflated_sharpe_ratio(r, n_trials=1, trial_sharpes_annual=[])
    assert ds.dsr == pytest.approx(probabilistic_sharpe_ratio(r))


def test_selection_bias_is_caught():
    trials = [noise(seed=s) for s in range(50)]          # nessuna ha un vantaggio vero
    sharpes = [sharpe_ratio(t) for t in trials]
    best = trials[int(np.argmax(sharpes))]
    ds = deflated_sharpe_ratio(best, n_trials=50, trial_sharpes_annual=sharpes)
    assert ds.psr_zero > 0.95      # preso da solo sembra "significativo"...
    assert ds.dsr < 0.95           # ...ma non dopo aver contato i 50 tentativi


def test_genuine_edge_survives_deflation():
    # Servono 10 anni: con 5 anni anche uno Sharpe di 2 scelto tra 21 tentativi
    # non basta (DSR ≈ 0.93), perché la soglia attesa per caso è ≈ 1.35.
    trials = [noise(n=2520, seed=s) for s in range(20)]
    real = noise(n=2520, mu=0.0012, seed=99)              # Sharpe annuo vero ≈ 1.9
    sharpes = [sharpe_ratio(t) for t in trials] + [sharpe_ratio(real)]
    ds = deflated_sharpe_ratio(real, n_trials=21, trial_sharpes_annual=sharpes)
    assert ds.dsr > 0.95


# --- Differenza tra Sharpe -----------------------------------------------------------

def test_identical_strategies_have_no_difference():
    r = noise(mu=0.0003)
    d = sharpe_difference_test(r, r)
    assert d.z == pytest.approx(0.0, abs=1e-9)
    assert d.p_value == pytest.approx(0.5)


def test_clearly_better_correlated_strategy_is_significant():
    b = noise(mu=0.0002, seed=2)
    a = b + 0.0004 + noise(sigma=0.001, seed=3)           # stessa esposizione + vantaggio costante
    assert sharpe_difference_test(a, b).p_value < 0.01


def test_equal_independent_strategies_not_significant():
    a, b = noise(mu=0.0003, seed=4), noise(mu=0.0003, seed=5)
    assert sharpe_difference_test(a, b).p_value > 0.05


# --- Walk-forward --------------------------------------------------------------------

def toy(prices: pd.DataFrame, invested: bool) -> pd.DataFrame:
    """Strategia giocattolo: tutto investito o tutto liquidità, ribilanciata ogni giorno."""
    return pd.DataFrame(1.0 if invested else 0.0, index=prices.index, columns=prices.columns)


@pytest.fixture
def rising_prices() -> pd.DataFrame:
    r = noise(n=252 * 12, mu=0.0006, seed=11)
    return pd.DataFrame({"A": 100 * (1 + r).cumprod()})


def test_walk_forward_folds_are_chronological(rising_prices):
    wf = walk_forward(rising_prices, toy, [{"invested": False}, {"invested": True}],
                      train_years=3, test_years=1)
    f = wf.folds
    assert (pd.to_datetime(f["train_end"]) < pd.to_datetime(f["test_start"])).all()
    assert (pd.to_datetime(f["test_start"]).iloc[1:].values
            > pd.to_datetime(f["test_end"]).iloc[:-1].values).all()
    assert wf.returns.index[0] >= pd.Timestamp(f["test_start"].iloc[0])


def test_walk_forward_picks_better_params(rising_prices):
    wf = walk_forward(rising_prices, toy, [{"invested": False}, {"invested": True}],
                      train_years=3, test_years=1)
    assert all(p == {"invested": True} for p in wf.folds["params"])


def test_walk_forward_no_leakage(rising_prices):
    grid = [{"invested": False}, {"invested": True}]
    base = walk_forward(rising_prices, toy, grid, train_years=3, test_years=1)
    last_test = pd.Timestamp(base.folds["test_start"].iloc[-1])
    crashed = rising_prices.copy()
    crashed.loc[last_test:] *= np.linspace(1, 0.3, len(crashed.loc[last_test:]))[:, None]
    alt = walk_forward(crashed, toy, grid, train_years=3, test_years=1)
    # La scelta per l'ultimo anno non può dipendere dal crollo dell'ultimo anno
    assert list(alt.folds["params"]) == list(base.folds["params"])
    pd.testing.assert_series_equal(base.returns.loc[: last_test - pd.Timedelta(days=1)],
                                   alt.returns.loc[: last_test - pd.Timedelta(days=1)])


# --- Criteri di promozione --------------------------------------------------------------

def test_noise_strategy_is_rejected(tmp_path):
    rng = np.random.default_rng(21)
    idx = pd.bdate_range("2005-01-01", "2022-12-30")
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (len(idx), 2)), axis=0),
                          index=idx, columns=["AAA", "BBB"])
    reg = ExperimentRegistry(tmp_path / "exp.sqlite")
    spec = ExperimentSpec(family="noise", strategy="time_series_momentum",
                          params={"lookback": 252}, universe=("AAA", "BBB"))
    bench = ExperimentSpec(family="benchmark", strategy="buy_and_hold",
                           params={"rebalance": "M"}, universe=("AAA", "BBB"))
    report = validate_candidate(spec, bench, reg, param_grid=[{"lookback": 126}, {"lookback": 252}],
                                prices=prices)
    assert report.verdict == "RESPINTA"
    assert {"Deflated Sharpe", "Sharpe > benchmark", "Walk-forward", "Stabilità",
            "Max drawdown"} <= set(report.table().index)
    assert report.details["run_id"] is not None   # il candidato è registrato


def test_criteria_are_immutable():
    with pytest.raises(Exception):
        Criteria().min_dsr = 0.5
