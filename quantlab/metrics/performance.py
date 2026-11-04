"""Metriche di performance calcolate da una serie di rendimenti periodici.

Convenzioni (valgono per tutto QuantLab):

- L'input è una `pd.Series` di **rendimenti semplici** per periodo
  (r_t = P_t / P_{t-1} - 1), già al netto dei costi se si tratta di una strategia.
- `periods_per_year` = 252 per dati giornalieri (giorni di borsa in un anno).
- Il drawdown è un numero **negativo** (es. -0.55 = -55% dal massimo).
- Se una metrica non è definita (es. volatilità zero) restituiamo NaN,
  mai un numero inventato.

Nota sull'annualizzazione: moltiplicare per sqrt(252) presuppone rendimenti
indipendenti nel tempo. Con rendimenti autocorrelati (frequenti in strategie
trend-following) lo Sharpe annualizzato può essere distorto (Lo, 2002,
"The Statistics of Sharpe Ratios", Financial Analysts Journal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Sotto questa dev. std. per periodo la volatilità è considerata nulla.
_ZERO_VOL = 1e-12


def _clean(returns: pd.Series) -> pd.Series:
    """Rimuove i NaN (es. il primo valore di pct_change)."""
    return pd.Series(returns, dtype="float64").dropna()


def returns_from_prices(prices: pd.Series) -> pd.Series:
    """Rendimenti semplici da una serie di prezzi (il primo valore viene scartato)."""
    return prices.pct_change().dropna()


def equity_curve(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    """Valore cumulato del capitale: start_value * prod(1 + r)."""
    return start_value * (1 + _clean(returns)).cumprod()


def total_return(returns: pd.Series) -> float:
    r = _clean(returns)
    if r.empty:
        return np.nan
    return float((1 + r).prod() - 1)


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Tasso di crescita annuo composto: (1 + R_tot)^(1/anni) - 1."""
    r = _clean(returns)
    if r.empty:
        return np.nan
    growth = float((1 + r).prod())
    if growth <= 0:
        return -1.0  # capitale azzerato
    years = len(r) / periods_per_year
    return growth ** (1 / years) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def _per_period_rate(annual_rate: float, periods_per_year: int) -> float:
    """Converte un tasso annuo in tasso per periodo (composto)."""
    return (1 + annual_rate) ** (1 / periods_per_year) - 1


def sharpe_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Sharpe ratio annualizzato: media(r - rf) / dev.std(r - rf) * sqrt(N).

    `risk_free` è un tasso **annuo** (es. 0.04 = 4%).
    """
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    excess = r - _per_period_rate(risk_free, periods_per_year)
    sd = excess.std(ddof=1)
    # Con rendimenti costanti la dev. std. dovrebbe essere 0, ma l'aritmetica
    # in virgola mobile lascia un residuo (~1e-18) che darebbe Sharpe assurdi.
    if np.isnan(sd) or sd < _ZERO_VOL:
        return np.nan
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sharpe_standard_error(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Errore standard approssimato dello Sharpe annualizzato (Lo, 2002, caso i.i.d.).

    SE(SR_periodo) ≈ sqrt((1 + SR_periodo² / 2) / T), poi annualizzato con sqrt(N).
    Serve a ricordare che uno Sharpe stimato su pochi anni è molto incerto:
    con 5 anni di dati giornalieri e Sharpe 0.5, l'errore standard è ~0.45.
    """
    r = _clean(returns)
    sr_annual = sharpe_ratio(r, risk_free, periods_per_year)
    if np.isnan(sr_annual):
        return np.nan
    sr_period = sr_annual / np.sqrt(periods_per_year)
    se_period = np.sqrt((1 + 0.5 * sr_period**2) / len(r))
    return float(se_period * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, target: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Sortino ratio: penalizza solo la volatilità "cattiva" (sotto il target).

    Deviazione al ribasso = sqrt(media(min(r - target, 0)²)) su *tutti* i periodi.
    `target` è un rendimento per periodo (default 0).
    """
    r = _clean(returns)
    if r.empty:
        return np.nan
    downside = np.minimum(r - target, 0.0)
    dd = np.sqrt((downside**2).mean())
    if dd < _ZERO_VOL:
        return np.nan
    return float((r.mean() - target) * periods_per_year / (dd * np.sqrt(periods_per_year)))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Distanza percentuale dal massimo precedente (0 = sui massimi).

    Il capitale iniziale (1.0) conta come primo massimo: se la strategia perde
    il primo giorno, quello è già un drawdown.
    """
    equity = equity_curve(returns)
    peak = np.maximum(equity.cummax(), 1.0)
    return equity / peak - 1


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    return float(dd.min()) if not dd.empty else np.nan


def max_drawdown_duration(returns: pd.Series) -> int:
    """Numero massimo di periodi consecutivi passati sotto il massimo precedente."""
    underwater = drawdown_series(returns) < 0
    longest = current = 0
    for is_under in underwater:
        current = current + 1 if is_under else 0
        longest = max(longest, current)
    return longest


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """CAGR / |max drawdown|: quanto rendimento per unità di perdita peggiore."""
    mdd = max_drawdown(returns)
    if np.isnan(mdd) or mdd == 0:
        return np.nan
    return cagr(returns, periods_per_year) / abs(mdd)


def hit_rate(returns: pd.Series) -> float:
    """Quota di periodi con rendimento positivo.

    Attenzione: è calcolata sui *giorni*, non sui *trade*. Il win rate per trade
    arriverà con il backtester, che conosce gli ingressi e le uscite.
    """
    r = _clean(returns)
    r = r[r != 0]
    return float((r > 0).mean()) if not r.empty else np.nan


def profit_factor(returns: pd.Series) -> float:
    """Somma dei guadagni / somma delle perdite (sui periodi, non sui trade)."""
    r = _clean(returns)
    losses = -r[r < 0].sum()
    if losses == 0:
        return np.nan
    return float(r[r > 0].sum() / losses)


def excess_returns(returns: pd.Series, rf: pd.Series | None) -> pd.Series:
    """Rendimenti in eccesso rispetto al tasso privo di rischio giornaliero (allineati per data)."""
    r = _clean(returns)
    if rf is None:
        return r
    return r - rf.reindex(r.index).fillna(0.0)


def summarize(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
    rf: pd.Series | None = None,
) -> pd.Series:
    """Tutte le metriche principali in un'unica tabella.

    Con `rf` (tasso privo di rischio giornaliero, es. T-bill) Sharpe, errore
    standard e Sortino sono calcolati sui rendimenti in eccesso, come nella
    definizione originale; CAGR, volatilità e drawdown restano sui rendimenti totali.
    """
    r = _clean(returns)
    ex = excess_returns(r, rf)
    sharpe_rf = 0.0 if rf is not None else risk_free
    return pd.Series(
        {
            "start": r.index[0] if not r.empty else None,
            "end": r.index[-1] if not r.empty else None,
            "periods": len(r),
            "total_return": total_return(r),
            "cagr": cagr(r, periods_per_year),
            "volatility": annualized_volatility(r, periods_per_year),
            "sharpe": sharpe_ratio(ex, sharpe_rf, periods_per_year),
            "sharpe_se": sharpe_standard_error(ex, sharpe_rf, periods_per_year),
            "sortino": sortino_ratio(ex, periods_per_year=periods_per_year),
            "cash_rate": (1 + (r - ex).mean()) ** periods_per_year - 1,
            "max_drawdown": max_drawdown(r),
            "max_dd_duration": max_drawdown_duration(r),
            "calmar": calmar_ratio(r, periods_per_year),
            "hit_rate": hit_rate(r),
            "profit_factor": profit_factor(r),
        },
        dtype="object",
    )
