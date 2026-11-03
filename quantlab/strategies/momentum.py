"""Time-series momentum (trend following).

Fonte: Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum",
Journal of Financial Economics 104(2), 228-250.

Evidenza documentata: su 58 futures e forward liquidi (indici azionari,
obbligazioni, valute, materie prime), con dati fino al 2009, il rendimento
degli ultimi 12 mesi di un asset ha previsto positivamente il suo rendimento
del mese successivo.

Spiegazioni proposte nel paper: reazione iniziale insufficiente alle notizie
(under-reaction) seguita da reazione eccessiva ritardata (over-reaction).

Questa implementazione è una versione semplificata per ETF:
- segnale = segno del rendimento total-return degli ultimi `lookback` giorni;
- ogni asset riceve 1/N del capitale: lungo se il segnale è positivo,
  altrimenti liquidità (`long_only=True`) o corto (`long_only=False`);
- ribilanciamento mensile.

Differenze rispetto al paper (da testare come ipotesi separate):
- il paper usa il rendimento *in eccesso* rispetto al tasso privo di rischio;
- il paper dimensiona le posizioni per volatilità (target 40% annuo per asset);
- il paper usa futures, qui ETF (con costi e universo diversi).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.strategies.base import keep_rebalance_dates


def time_series_momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    long_only: bool = True,
    rebalance: str | None = "M",
) -> pd.DataFrame:
    """Pesi target della strategia time-series momentum.

    Args:
        prices: prezzi di chiusura aggiustati, una colonna per asset.
        lookback: finestra del rendimento passato, in giorni di borsa (252 ≈ 12 mesi).
        long_only: se True le posizioni sono lungo o liquidità, mai corte.
        rebalance: frequenza di ribilanciamento ("M" = mensile, None = giornaliero).
    """
    if lookback < 1:
        raise ValueError("lookback deve essere almeno 1")

    past_return = prices / prices.shift(lookback) - 1
    signal = np.sign(past_return)
    if long_only:
        signal = signal.clip(lower=0)

    weights = signal / prices.shape[1]
    # Durante il periodo iniziale (warm-up) nessun asset ha un segnale: la riga
    # resta tutta NaN e il motore non opera.
    return keep_rebalance_dates(weights, rebalance)
