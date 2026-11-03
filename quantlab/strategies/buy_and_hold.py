"""Buy-and-hold: il benchmark contro cui confrontare ogni strategia.

Se una strategia attiva non batte, al netto dei costi e in modo statisticamente
credibile, il semplice "compra e tieni" sugli stessi asset, non ha valore.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.strategies.base import keep_rebalance_dates


def buy_and_hold(prices: pd.DataFrame, rebalance: str | None = None) -> pd.DataFrame:
    """Pesi uguali su tutti gli asset, investiti quando tutti hanno un prezzo.

    Args:
        prices: prezzi di chiusura, una colonna per asset.
        rebalance: None = compra una volta e lascia derivare i pesi;
            "M" = riporta i pesi uguali ogni fine mese.
    """
    n = prices.shape[1]
    all_listed = prices.notna().all(axis=1)
    if not all_listed.any():
        raise ValueError("nessuna data in cui tutti gli asset hanno un prezzo")

    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    if rebalance is None:
        weights.loc[all_listed.idxmax()] = 1.0 / n
        return weights

    weights.loc[all_listed] = 1.0 / n
    first = keep_rebalance_dates(weights, rebalance)
    # Investi subito alla prima data valida, senza aspettare la fine del mese.
    first.loc[all_listed.idxmax()] = 1.0 / n
    return first
