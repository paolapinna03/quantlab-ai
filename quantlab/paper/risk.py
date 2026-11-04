"""Risk manager indipendente del paper trading.

È separato dalla strategia di proposito: la strategia propone dei pesi, il
risk manager decide se possono diventare ordini. Se un controllo fallisce,
gli ordini vengono bloccati e l'evento viene registrato con il motivo.

Regola di sicurezza: nel dubbio, non operare.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskLimits:
    max_weight: float = 0.45          # peso massimo di un singolo ETF
    max_gross: float = 1.0            # esposizione totale massima (niente leva)
    max_turnover: float = 2.0         # somma |variazioni di peso| per ribilanciamento (2 = rotazione completa)
    halt_drawdown: float = -0.20      # sotto questa perdita dal massimo: liquidazione e arresto
    max_data_age_days: int = 5        # giorni di calendario tra l'ultimo prezzo e oggi
    max_daily_move: float = 0.25      # rendimento giornaliero oltre il quale il dato è sospetto


def check_targets(targets: pd.Series, current: pd.Series, limits: RiskLimits) -> list[str]:
    """Problemi che impediscono di trasformare i pesi proposti in ordini (lista vuota = ok)."""
    problems = []
    t = targets.fillna(0.0)
    if (t < -1e-12).any():
        problems.append(f"pesi negativi non ammessi: {list(t.index[t < 0])}")
    if t.max() > limits.max_weight + 1e-9:
        problems.append(f"{t.idxmax()} al {t.max():.0%}, oltre il tetto del {limits.max_weight:.0%}")
    if t.abs().sum() > limits.max_gross + 1e-9:
        problems.append(f"esposizione {t.abs().sum():.0%}, oltre il {limits.max_gross:.0%}")
    turnover = (t - current.reindex(t.index).fillna(0.0)).abs().sum()
    if turnover > limits.max_turnover + 1e-9:
        problems.append(f"turnover {turnover:.2f}, oltre il limite di {limits.max_turnover:.2f}")
    return problems


def check_data(prices: pd.DataFrame, today: pd.Timestamp, limits: RiskLimits) -> tuple[list[str], list[str]]:
    """(problemi bloccanti, avvisi) sui dati di mercato appena scaricati."""
    blocking, warnings = [], []
    if prices.empty:
        return ["nessun prezzo disponibile"], warnings
    age = (pd.Timestamp(today).normalize() - prices.index[-1]).days
    if age > limits.max_data_age_days:
        blocking.append(f"ultimo prezzo del {prices.index[-1].date()}, vecchio di {age} giorni")
    last = prices.iloc[-1]
    missing = list(last.index[last.isna()])
    if missing:
        blocking.append(f"prezzo mancante oggi per {missing}")
    moves = prices.pct_change().iloc[-1].abs()
    jumps = moves[moves > limits.max_daily_move]
    for asset, m in jumps.items():
        warnings.append(f"{asset} si è mosso del {m:.0%} in un giorno: dato da verificare")
    return blocking, warnings


def drawdown(equity: float, peak: float) -> float:
    return equity / peak - 1 if peak > 0 else np.nan
