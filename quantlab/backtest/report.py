"""Valutazione di un backtest su una finestra temporale.

Strategie diverse vanno confrontate sulla **stessa finestra**: una strategia
con 12 mesi di warm-up non può essere confrontata con un buy-and-hold che
include anche quei 12 mesi.
"""

from __future__ import annotations

import pandas as pd

from quantlab.backtest.engine import BacktestResult
from quantlab.metrics import TRADING_DAYS, summarize


def evaluate(
    result: BacktestResult,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """Metriche di performance + turnover, costi ed esposizione media."""
    window = slice(start, end)
    s = summarize(result.returns.loc[window], periods_per_year=periods_per_year)
    years = s["periods"] / periods_per_year
    s["turnover_yr"] = result.turnover.loc[window].sum() / years
    s["cost_yr"] = result.costs.loc[window].sum() / years
    s["invested"] = result.weights.loc[window].abs().sum(axis=1).mean()
    return s


def warmup_end(prices: pd.DataFrame, lookback: int, buffer_days: int = 22) -> pd.Timestamp:
    """Prima data utile per valutare: tutti gli asset quotati + lookback + ~1 mese.

    Il mese di margine lascia alla strategia il tempo di arrivare al primo
    ribilanciamento dopo il warm-up.
    """
    all_listed = prices.notna().all(axis=1)
    if not all_listed.any():
        raise ValueError("nessuna data in cui tutti gli asset hanno un prezzo")
    first = prices.index.get_loc(all_listed.idxmax())
    pos = first + lookback + buffer_days
    if pos >= len(prices):
        raise ValueError("storia troppo corta per questo lookback")
    return prices.index[pos]
