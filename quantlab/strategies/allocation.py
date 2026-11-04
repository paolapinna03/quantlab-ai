"""Allocazione basata sul rischio, ricalcolata a ogni ribilanciamento.

A ogni data di ribilanciamento t:
1. si prendono i rendimenti degli ultimi `lookback` giorni FINO A t compreso;
2. si stima la covarianza (con shrinkage di Ledoit-Wolf, se richiesto);
3. si calcolano i pesi con il metodo scelto (vedi quantlab.portfolio);
4. facoltativo: gli asset con rendimento negativo negli ultimi `momentum_filter`
   giorni vengono azzerati e la loro quota resta in liquidità (come nel
   time-series momentum, ma con pesi basati sul rischio invece che 1/N).

Gli asset non ancora quotati per l'intera finestra vengono esclusi e
rientrano quando hanno abbastanza storia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.portfolio.optimizers import (
    METHODS,
    hrp,
    inverse_vol,
    ledoit_wolf,
    mean_variance,
    min_variance,
    risk_parity,
    sample_covariance,
)
from quantlab.strategies.base import period_end_mask


def risk_allocation(
    prices: pd.DataFrame,
    method: str = "risk_parity",
    lookback: int = 252,
    rebalance: str = "M",
    max_weight: float = 0.4,
    shrinkage: bool = True,
    momentum_filter: int = 0,
) -> pd.DataFrame:
    """Pesi target: una riga piena a ogni ribilanciamento, NaN negli altri giorni."""
    if method not in METHODS:
        raise ValueError(f"metodo sconosciuto: {method!r}. Disponibili: {METHODS}")
    if lookback < 20:
        raise ValueError("lookback troppo corto per stimare una covarianza (minimo 20)")

    rets = prices.pct_change()
    out = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    rebal = period_end_mask(prices.index, rebalance)
    for i in np.flatnonzero(rebal):
        # Si investe solo quando sia la covarianza sia il filtro momentum sono calcolabili:
        # investire prima, senza filtro, sarebbe una strategia diversa da quella dichiarata.
        if i < max(lookback, momentum_filter):
            continue
        window = rets.iloc[i - lookback + 1 : i + 1]
        cols = window.columns[window.notna().all()]
        if len(cols) == 0:
            continue
        x = window[cols].to_numpy()
        cov = ledoit_wolf(x)[0] if shrinkage else sample_covariance(x)
        if method == "inverse_vol":
            w = inverse_vol(cov)
        elif method == "min_variance":
            w = min_variance(cov, max_weight)
        elif method == "risk_parity":
            w = risk_parity(cov)
        elif method == "hrp":
            w = hrp(cov)
        else:
            w = mean_variance(x.mean(axis=0), cov, max_weight)

        row = pd.Series(0.0, index=prices.columns)
        row[cols] = w
        if momentum_filter > 0:
            past = prices.iloc[i] / prices.iloc[i - momentum_filter] - 1
            row[~(past > 0)] = 0.0          # trend negativo o dato mancante → liquidità
        out.iloc[i] = row.to_numpy()
    return out
