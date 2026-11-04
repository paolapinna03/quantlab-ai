"""Trend su più orizzonti e volatility targeting.

Due estensioni del time-series momentum (vedi `momentum.py`):

1. **Trend su più orizzonti.** Invece di un solo rendimento passato (12 mesi),
   per ogni asset si guarda il segno del rendimento su più finestre e si fa la
   media. Con orizzonti 1, 3, 6 e 12 mesi un asset può essere investito al 0%,
   25%, 50%, 75% o 100% della sua quota 1/N. Riduce la dipendenza da una sola
   finestra e reagisce prima alle inversioni.
   Fonti: Hurst, Ooi & Pedersen (2017), "A Century of Evidence on Trend-Following
   Investing", Journal of Portfolio Management; Levine & Pedersen (2016).

2. **Volatility targeting.** A ogni ribilanciamento si stima la volatilità del
   portafoglio con i rendimenti degli ultimi `vol_lookback` giorni. Se supera
   `vol_target`, tutte le posizioni vengono ridotte in proporzione e il resto va
   in liquidità. Mai leva: l'esposizione non supera mai il 100%.
   Fonte: Moreira & Muir (2017), "Volatility-Managed Portfolios", Journal of
   Finance 72(4).

Regola anti look-ahead: i pesi della data t usano solo prezzi fino alla chiusura
di t; il motore li esegue il giorno successivo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.strategies.base import period_end_mask


def trend_signals(
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (252,),
    vol_target: float = 0.0,
    vol_lookback: int = 63,
    rebalance: str = "M",
) -> pd.DataFrame:
    """Pesi target: una riga a ogni ribilanciamento, NaN negli altri giorni.

    Args:
        prices: prezzi di chiusura aggiustati, una colonna per asset.
        horizons: finestre del rendimento passato in giorni di borsa (21 ≈ 1 mese).
        vol_target: volatilità annua massima del portafoglio (0 = nessun limite).
        vol_lookback: giorni usati per stimare la volatilità.
        rebalance: frequenza di ribilanciamento ("M" = mensile).
    """
    horizons = tuple(int(h) for h in horizons)
    if not horizons or min(horizons) < 1:
        raise ValueError("servono uno o più orizzonti, ciascuno di almeno 1 giorno")
    if vol_target < 0:
        raise ValueError("vol_target non può essere negativo")
    if vol_target > 0 and vol_lookback < 20:
        raise ValueError("vol_lookback troppo corto per stimare la volatilità (minimo 20)")

    n = prices.shape[1]
    rets = prices.pct_change()
    warmup = max(max(horizons), vol_lookback if vol_target > 0 else 0)
    out = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for i in np.flatnonzero(period_end_mask(prices.index, rebalance)):
        if i < warmup:
            continue
        signals = []
        for h in horizons:
            past = prices.iloc[i] / prices.iloc[i - h] - 1
            signals.append((past > 0).astype(float).where(past.notna()))
        signal = pd.concat(signals, axis=1).mean(axis=1, skipna=False)   # NaN se manca storia
        w = (signal / n).fillna(0.0)
        if vol_target > 0 and w.sum() > 0:
            window = rets.iloc[i - vol_lookback + 1 : i + 1]
            held = w.index[(w > 0) & window.notna().all()]
            cov = window[held].cov().to_numpy() * 252
            vol = float(np.sqrt(w[held].to_numpy() @ cov @ w[held].to_numpy()))
            if vol > vol_target:
                w = w * (vol_target / vol)
        out.iloc[i] = w.to_numpy()
    return out
