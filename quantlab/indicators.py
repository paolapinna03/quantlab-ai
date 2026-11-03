"""Indicatori tecnici, tutti causali.

Regola: il valore alla data t usa solo dati fino alla chiusura di t.
Dove serve escludere la barra corrente (es. canale di Donchian per un
breakout) lo diciamo esplicitamente nel nome o nella docstring.
"""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Media mobile semplice sugli ultimi `window` valori (inclusa la data t)."""
    return series.rolling(window, min_periods=window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index di Wilder (0-100).

    Usa la media esponenziale di Wilder (alpha = 1/period). Valori bassi =
    il prezzo è sceso molto di recente ("ipervenduto").
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    # Solo rialzi nella finestra: avg_loss = 0 → RSI = 100
    out = out.where(avg_loss != 0, 100.0)
    return out.where(avg_gain.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range di Wilder: ampiezza media delle barre, misura di volatilità in prezzo."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def donchian_high_prior(high: pd.Series, window: int) -> pd.Series:
    """Massimo dei `window` giorni PRECEDENTI (esclusa la data t).

    Per un breakout: "la chiusura di oggi supera il massimo degli ultimi N giorni".
    Includere oggi renderebbe il confronto impossibile da superare.
    """
    return high.rolling(window, min_periods=window).max().shift(1)


def donchian_low_prior(low: pd.Series, window: int) -> pd.Series:
    """Minimo dei `window` giorni PRECEDENTI (esclusa la data t)."""
    return low.rolling(window, min_periods=window).min().shift(1)
