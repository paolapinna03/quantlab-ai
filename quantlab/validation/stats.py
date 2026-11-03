"""Statistiche per giudicare se uno Sharpe ratio è credibile.

Fonti:
- Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier",
  Journal of Risk 15(2): Probabilistic Sharpe Ratio (PSR).
- Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality",
  Journal of Portfolio Management 40(5): Deflated Sharpe Ratio (DSR).
- Jobson & Korkie (1981) con la correzione di Memmel (2003), "Performance
  hypothesis testing with the Sharpe ratio", Finance Letters: test sulla
  differenza tra due Sharpe ratio.

Convenzione: dentro le formule lo Sharpe è **per periodo** (giornaliero, non
annualizzato). Le funzioni accettano e restituiscono valori annualizzati solo
dove il nome lo dice esplicitamente.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from quantlab.metrics import TRADING_DAYS

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def _clean(returns: pd.Series) -> pd.Series:
    return pd.Series(returns, dtype="float64").dropna()


def _per_period_sharpe(r: pd.Series) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 1e-12 else np.nan


def probabilistic_sharpe_ratio(
    returns: pd.Series, sr_benchmark_annual: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Probabilità che lo Sharpe *vero* superi `sr_benchmark_annual`.

    Tiene conto della lunghezza del campione e della non-normalità dei rendimenti:
    asimmetria negativa e code spesse riducono la fiducia nello Sharpe stimato.

        PSR = Φ( (SR - SR*) · √(T-1) / √(1 - γ₃·SR + (γ₄-1)/4 · SR²) )
    """
    r = _clean(returns)
    if len(r) < 3:
        return np.nan
    sr = _per_period_sharpe(r)
    if np.isnan(sr):
        return np.nan
    sr_star = sr_benchmark_annual / math.sqrt(periods_per_year)
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0  # pandas restituisce la curtosi in eccesso
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0:
        return np.nan
    z = (sr - sr_star) * math.sqrt(len(r) - 1) / math.sqrt(denom)
    return _NORMAL.cdf(z)


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Sharpe massimo atteso tra `n_trials` strategie *senza alcun vantaggio reale*.

    Se provi N varianti tutte con Sharpe vero zero, la migliore avrà comunque
    uno Sharpe stimato positivo. Questa è la soglia da superare.
    `sr_std` = deviazione standard degli Sharpe stimati tra i tentativi.
    """
    if n_trials <= 1:
        return 0.0
    n = n_trials
    return sr_std * (
        (1 - EULER_GAMMA) * _NORMAL.inv_cdf(1 - 1 / n)
        + EULER_GAMMA * _NORMAL.inv_cdf(1 - 1 / (n * math.e))
    )


@dataclass
class DeflatedSharpe:
    dsr: float                  # probabilità che lo Sharpe vero superi la soglia di selezione
    psr_zero: float             # PSR rispetto a zero, senza correzione per i tentativi
    sharpe_annual: float
    threshold_annual: float     # Sharpe massimo atteso per caso, annualizzato
    n_trials: int
    note: str = ""


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    trial_sharpes_annual: list[float],
    periods_per_year: int = TRADING_DAYS,
) -> DeflatedSharpe:
    """Deflated Sharpe Ratio: il PSR rispetto allo Sharpe atteso dal miglior tentativo casuale.

    Args:
        returns: rendimenti della strategia scelta.
        n_trials: numero di configurazioni distinte provate (dal registro).
        trial_sharpes_annual: Sharpe annualizzati dei tentativi, per stimarne la dispersione.
    """
    r = _clean(returns)
    sqrt_n = math.sqrt(periods_per_year)
    sharpes = [s / sqrt_n for s in trial_sharpes_annual if s is not None and not np.isnan(s)]
    note = ""
    if n_trials <= 1:
        sr0 = 0.0
        note = "un solo tentativo: DSR coincide con PSR(0)"
    elif len(sharpes) < 2:
        sr0 = np.nan
        note = "servono almeno 2 Sharpe registrati per stimarne la dispersione"
    else:
        sr0 = expected_max_sharpe(n_trials, float(np.std(sharpes, ddof=1)))
        if len(sharpes) < 10:
            note = f"dispersione stimata su soli {len(sharpes)} tentativi: stima rumorosa"

    psr0 = probabilistic_sharpe_ratio(r, 0.0, periods_per_year)
    dsr = probabilistic_sharpe_ratio(r, sr0 * sqrt_n, periods_per_year) if not np.isnan(sr0) else np.nan
    sr = _per_period_sharpe(r)
    return DeflatedSharpe(
        dsr=dsr, psr_zero=psr0, sharpe_annual=sr * sqrt_n,
        threshold_annual=sr0 * sqrt_n, n_trials=n_trials, note=note,
    )


@dataclass
class SharpeDifference:
    sharpe_a_annual: float
    sharpe_b_annual: float
    z: float
    p_value: float              # unilaterale: H0 = Sharpe(a) <= Sharpe(b)
    correlation: float


def sharpe_difference_test(
    returns_a: pd.Series, returns_b: pd.Series, periods_per_year: int = TRADING_DAYS
) -> SharpeDifference:
    """La strategia `a` ha uno Sharpe significativamente maggiore di `b`?

    Test di Jobson-Korkie con correzione di Memmel (2003), sugli stessi giorni:

        Var(SR_a - SR_b) ≈ [2(1-ρ) + ½(SR_a² + SR_b² - 2·SR_a·SR_b·ρ²)] / T

    Ipotesi: rendimenti i.i.d. e circa normali. Con code spesse il test è
    ottimista; va letto insieme agli altri controlli.
    """
    a, b = _clean(returns_a).align(_clean(returns_b), join="inner")
    t = len(a)
    sa, sb = _per_period_sharpe(a), _per_period_sharpe(b)
    rho = float(np.corrcoef(a, b)[0, 1])
    var = (2 * (1 - rho) + 0.5 * (sa**2 + sb**2 - 2 * sa * sb * rho**2)) / t
    if var > 1e-18:
        z = (sa - sb) / math.sqrt(var)
    elif abs(sa - sb) < 1e-12:
        z = 0.0      # serie identiche: nessuna differenza
    else:
        z = math.copysign(math.inf, sa - sb)
    p = 1 - _NORMAL.cdf(z) if math.isfinite(z) else (0.0 if z > 0 else 1.0)
    sqrt_n = math.sqrt(periods_per_year)
    return SharpeDifference(sa * sqrt_n, sb * sqrt_n, z, p, rho)
