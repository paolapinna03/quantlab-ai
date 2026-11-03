"""Controlli di qualità sui dati OHLCV.

Un backtest non è mai più affidabile dei dati che usa. Qui distinguiamo:

- **errori**: rendono i dati inutilizzabili (es. date duplicate, prezzi <= 0).
  Il caricamento si ferma.
- **avvisi**: anomalie da conoscere ma non necessariamente sbagliate
  (es. un rendimento giornaliero del -20% può essere un crollo reale).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Tolleranza relativa per i controlli high/low: i prezzi "adjusted" di Yahoo
# vengono riscalati e possono violare di pochissimo high >= close, ecc.
_PRICE_TOL = 1e-4

# Soglie per gli avvisi.
EXTREME_RETURN = 0.25   # |rendimento giornaliero| oltre il 25%
MAX_GAP_DAYS = 7        # giorni di calendario senza dati (festività: max ~4)


class DataValidationError(ValueError):
    """I dati non superano i controlli minimi e non vanno usati."""


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self, label: str = "") -> None:
        if self.errors:
            prefix = f"{label}: " if label else ""
            raise DataValidationError(prefix + "; ".join(self.errors))


def validate_ohlcv(df: pd.DataFrame) -> ValidationReport:
    """Controlla un DataFrame OHLCV giornaliero e restituisce errori e avvisi."""
    report = ValidationReport()

    # --- Struttura -------------------------------------------------------
    if df.empty:
        report.errors.append("nessun dato")
        return report

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        report.errors.append(f"colonne mancanti: {missing}")
        return report

    if not isinstance(df.index, pd.DatetimeIndex):
        report.errors.append("l'indice non è di tipo data")
        return report

    if df.index.has_duplicates:
        n = int(df.index.duplicated().sum())
        report.errors.append(f"{n} date duplicate")

    if not df.index.is_monotonic_increasing:
        report.errors.append("date non in ordine crescente")

    # --- Valori ----------------------------------------------------------
    prices = df[["open", "high", "low", "close"]]

    n_nan_close = int(df["close"].isna().sum())
    if n_nan_close:
        report.errors.append(f"{n_nan_close} prezzi di chiusura mancanti")

    if (prices <= 0).any().any():
        report.errors.append("prezzi <= 0")

    if (df["volume"] < 0).any():
        report.errors.append("volumi negativi")

    # Coerenza OHLC: il massimo deve essere >= di tutto, il minimo <= di tutto.
    hi_ok = df["high"] * (1 + _PRICE_TOL) >= prices[["open", "close", "low"]].max(axis=1)
    lo_ok = df["low"] * (1 - _PRICE_TOL) <= prices[["open", "close", "high"]].min(axis=1)
    n_bad = int((~(hi_ok & lo_ok)).sum())
    if n_bad:
        report.warnings.append(f"{n_bad} barre con high/low incoerenti")

    # --- Anomalie (avvisi) ----------------------------------------------
    returns = df["close"].pct_change().dropna()
    extreme = returns[np.abs(returns) > EXTREME_RETURN]
    for date, r in extreme.items():
        report.warnings.append(f"rendimento estremo {r:+.1%} il {date.date()}")

    gaps = df.index.to_series().diff().dt.days
    for date, days in gaps[gaps > MAX_GAP_DAYS].items():
        report.warnings.append(f"buco di {int(days)} giorni prima del {date.date()}")

    n_zero_vol = int((df["volume"] == 0).sum())
    if n_zero_vol:
        report.warnings.append(f"{n_zero_vol} giorni con volume zero")

    return report
