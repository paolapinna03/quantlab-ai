"""Strumenti comuni alle strategie.

Una strategia in QuantLab è una funzione:

    prezzi di chiusura (DataFrame)  →  pesi target (DataFrame)

con la regola fondamentale: **il peso della data t può usare solo dati fino
alla chiusura di t**. Il motore di backtest si occupa del ritardo di esecuzione.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def period_end_mask(index: pd.DatetimeIndex, freq: str = "M") -> np.ndarray:
    """True nell'ultimo giorno di borsa di ogni periodo (mese "M", trimestre "Q"...).

    Sapere che domani inizia un nuovo mese non è look-ahead: il calendario di
    borsa è pubblicato in anticipo. L'ultima data della serie non è mai una
    fine periodo, perché non sappiamo se il periodo è davvero finito.
    """
    periods = pd.Series(index.to_period(freq))
    mask = (periods != periods.shift(-1)).to_numpy().copy()
    if len(mask):
        mask[-1] = False
    return mask


def keep_rebalance_dates(weights: pd.DataFrame, freq: str | None) -> pd.DataFrame:
    """Mantiene i pesi solo nelle date di ribilanciamento (le altre diventano NaN)."""
    if freq is None:
        return weights
    mask = period_end_mask(weights.index, freq)
    out = weights.copy()
    out.loc[~mask] = np.nan
    return out
